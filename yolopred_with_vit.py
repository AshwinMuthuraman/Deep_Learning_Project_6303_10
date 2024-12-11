import streamlit as st
import cv2
import numpy as np
import torch
import torchvision  # For NMS
from PIL import Image as PILImage  # Aliased to prevent conflicts
import albumentations as A
from albumentations.pytorch import ToTensorV2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import uuid
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt  # Import matplotlib for grid visualization

# ---------------------------- Constants and Configurations ----------------------------

# YOLOv7 Configuration
YOLO_WEIGHTS_PATH = "/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/best.pt"  # Path to YOLO model weights
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_CLASS_NAMES = ["chessboard"]  # List of class names

# FEN Model Configuration
FEN_MODEL_PATH = '/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/Deep_Learning_Project_6303_10/VIT_early_stop_best_model.pth'  # Path to FEN model weights
FEN_IMAGE_SIZE = 224
LABEL_TO_FEN_SHORT = {
    0: '0', 1: 'p', 2: 'P',
    3: 'b', 4: 'B',
    5: 'r', 6: 'R',
    7: 'n', 8: 'N',
    9: 'q', 10: 'Q',
    11: 'k', 12: 'K'
}

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Supported Image Extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# RTC Configuration for streamlit-webrtc
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------------------------- YOLOv7 Functions ----------------------------

@st.cache_resource
def load_yolo_model(weights_path, device):
    """
    Loads the YOLOv7 model from the provided weights file.
    """
    try:
        model = torch.load(weights_path, map_location=device)['model'].float().eval()
        st.success("YOLOv7 model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv7 model: {e}")
        return None

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Filters YOLO predictions by confidence and performs Non-Maximum Suppression (NMS).
    """
    output = []
    for image_pred in prediction:  # Per image
        # Filter by confidence
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        if not image_pred.shape[0]:
            continue

        # Multiply confidence by class probability
        image_pred[:, 5:] *= image_pred[:, 4:5]

        # Get boxes with score and class
        boxes = xywh2xyxy(image_pred[:, :4])
        scores, class_ids = image_pred[:, 5:].max(1)
        detections = torch.cat((boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)), dim=1)

        # Perform NMS
        keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thres)
        output.append(detections[keep])
    return output

def xywh2xyxy(x):
    """
    Convert YOLOv7 format from center x, center y, width, height to x1, y1, x2, y2
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def rescale_coords(img_shape, coords, original_shape):
    """
    Rescale coordinates from model to image space
    """
    gain = min(img_shape[0] / original_shape[0], img_shape[1] / original_shape[1])  # Gain
    pad = (img_shape[1] - original_shape[1] * gain) / 2, (img_shape[0] - original_shape[0] * gain) / 2  # Padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, original_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, original_shape[0])  # y1, y2
    return coords

def run_yolo_inference(model, image, conf_thres=0.25, iou_thres=0.45, device='cuda'):
    """
    Perform inference on an image using YOLOv7.
    """
    original_shape = image.shape[:2]  # Height, Width

    # Preprocess Image
    img = letterbox(image, new_shape=(640, 640))[0]  # Resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Run Inference
    with torch.no_grad():
        try:
            pred = model(img)[0]  # Run model
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
        except Exception as e:
            st.error(f"Error during YOLO inference: {e}")
            return []

    # Process Predictions
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = rescale_coords(img.shape[2:], det[:, :4], original_shape).round()
            detections.extend(det.cpu().numpy())  # Convert to numpy

    return detections

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and pads an image to fit the desired shape.
    """
    shape = img.shape[:2]  # Current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Ratio  = new / old
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # W, H
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # Padding
    dw, dh = dw // 2, dh // 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def draw_yolo_boxes(img, detections, class_names):
    """
    Draws bounding boxes and labels on the image.
    """
    for *xyxy, conf, cls in detections:
        label = f"{class_names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img

# ---------------------------- Chessboard Processing Functions ----------------------------

def process_chessboard(img, bbox, fixed_grid_size=(640, 640)):
    """
    Processes the detected chessboard bounding box to find corners and apply perspective transformation.
    Returns the image with drawn corners and the warped (aligned) chessboard image.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except ValueError as ve:
        st.warning(f"Invalid bounding box coordinates: {ve}")
        return img, None

    if y2 <= y1 or x2 <= x1:
        st.warning("Invalid bounding box dimensions.")
        return img, None

    chessboard_roi = img[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(chessboard_roi, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.warning("No contours found in the chessboard region.")
        return img, None

    # Select the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Find corners of the largest contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) != 4:
        st.warning("Detected contour does not have 4 corners.")
        return img, None

    # Order the corners in a consistent order: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [fixed_grid_size[0] - 1, 0],
        [fixed_grid_size[0] - 1, fixed_grid_size[1] - 1],
        [0, fixed_grid_size[1] - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(chessboard_roi, M, fixed_grid_size)
    except cv2.error as e:
        st.warning(f"Perspective transformation failed: {e}")
        return img, None

    # Draw the detected corners on the original image for visualization
    for point in rect:
        cv2.circle(img, (x1 + int(point[0]), y1 + int(point[1])), 5, (0, 255, 0), -1)

    return img, warped

def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find top-left and bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

# ---------------------------- FEN Prediction Functions ----------------------------

@st.cache_resource
def load_fen_model(model_path, num_classes=13):
    """
    Load the trained FEN model with your saved weights.
    """
    try:
        model = define_fen_model(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.success("FEN prediction model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading FEN model: {e}")
        return None

def define_fen_model(num_classes=13):
    """
    Define your model architecture. Ensure it matches the architecture used for training.
    """
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)  # Do not load pretrained weights
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_ftrs, num_classes)  # Match your training setup
    return model

def split_image_into_grids(image):
    """
    Splits a chessboard image into 64 grids (8x8).
    Handles both grayscale and RGB images.
    """
    h, w = image.shape[:2]  # Extract only height and width
    grid_h, grid_w = h // 8, w // 8  # Dynamically calculate grid size
    grids = []
    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            grid = image[i:i + grid_h, j:j + grid_w]
            grids.append(grid)
    return grids

def label_to_fen(label_list):
    """
    Converts a list of labels (0-12) into a valid FEN notation string.
    """
    fen = ''
    empty_count = 0
    for idx, label in enumerate(label_list):
        if label == 0:
            empty_count += 1
        else:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            fen += LABEL_TO_FEN_SHORT.get(label, '0')
        if (idx + 1) % 8 == 0:  # End of a row
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            if idx != len(label_list) - 1:  # Add "/" between rows
                fen += '/'
    return fen

# Transform for preprocessing images
transform = A.Compose([
    A.Resize(height=FEN_IMAGE_SIZE, width=FEN_IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_fen_from_image(model, image):
    """
    Predicts the FEN notation for a given chessboard image.
    The image should be a BGR or RGB image as a NumPy array.
    """
    if image is None:
        st.error("No image provided for FEN prediction.")
        return "Error"

    # Convert to grayscale if not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    # Split image into grids
    grids = split_image_into_grids(image_gray)
    if len(grids) != 64:
        st.error(f"Image does not have 64 grids (got {len(grids)}). Check cropping and warping.")
        return "Error"

    # Predict the labels for each grid
    predicted_labels = []
    for grid_idx, grid in enumerate(grids, 1):
        try:
            # Preprocess grid
            grid_resized = cv2.resize(grid, (FEN_IMAGE_SIZE, FEN_IMAGE_SIZE))
            grid_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2RGB)
            input_tensor = transform(image=np.array(grid_rgb))['image'].unsqueeze(0).to(DEVICE)

            # Model prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_labels.append(pred.item())
        except Exception as e:
            st.warning(f"Error processing grid {grid_idx}: {e}")
            predicted_labels.append(0)  # Assuming '0' is 'empty'

    # Convert labels to FEN
    fen = label_to_fen(predicted_labels)
    return fen

def visualize_grids(image, title="Chessboard Grids"):
    """
    Visualizes the 64 grids of the chessboard for debugging purposes.
    """
    grids = split_image_into_grids(image)
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(grids[i], cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------- Utility Functions ----------------------------

def sanitize_fen(fen):
    """
    Sanitizes the FEN string to be used as a filename by replacing or removing invalid characters.
    """
    # Replace '/' with '_', as '/' is not allowed in filenames
    sanitized = fen.replace('/', '_')
    # Remove or replace other characters if necessary
    # For simplicity, we'll replace any remaining non-alphanumeric characters with '_'
    sanitized = ''.join([c if c.isalnum() else '_' for c in sanitized])
    return sanitized

# ---------------------------- Webcam Video Processor ----------------------------

class VideoProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, fen_model):
        self.yolo_model = yolo_model
        self.fen_model = fen_model
        self.fen_results = []  # Initialize as an empty list

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        original_img = img.copy()

        # YOLOv7 Inference
        detections = run_yolo_inference(self.yolo_model, img, conf_thres=YOLO_CONF_THRESHOLD, 
                                        iou_thres=YOLO_IOU_THRESHOLD, device=DEVICE)

        # Draw Bounding Boxes
        annotated_image = draw_yolo_boxes(img.copy(), detections, YOLO_CLASS_NAMES)

        # Process Each Detection
        for det_idx, det in enumerate(detections, 1):
            cls = int(det[5])
            if YOLO_CLASS_NAMES[cls] != "chessboard":
                continue  # Skip if not a chessboard

            # Apply the corner detection and perspective transform pipeline
            processed_img, warped = process_chessboard(img, det)
            if warped is not None:
                # Predict FEN
                fen = predict_fen_from_image(self.fen_model, warped)
                self.fen_results.append(fen)

                # Overlay FEN annotation near the bounding box on the annotated image
                x1, y1, x2, y2 = map(int, det[:4])
                # Position the text slightly above the top-left corner of the bounding box
                text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                cv2.putText(annotated_image, fen, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Limit the number of stored FEN results to the last 10
                if len(self.fen_results) > 10:
                    self.fen_results.pop(0)

        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# ---------------------------- Streamlit Application ----------------------------

def main():
    st.title("Chessboard Detection and FEN Prediction")
    st.write("Upload an image or use your webcam to detect chessboards and generate FEN notation.")

    # Load Models
    yolo_model = load_yolo_model(YOLO_WEIGHTS_PATH, DEVICE)
    fen_model = load_fen_model(FEN_MODEL_PATH)

    if yolo_model is None or fen_model is None:
        st.error("Failed to load the necessary models. Please check the model paths.")
        return

    # Sidebar for Options
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Input Method", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        uploaded_files = st.sidebar.file_uploader("Upload Image(s)", type=IMAGE_EXTENSIONS, accept_multiple_files=True)
        if uploaded_files:
            for img_idx, uploaded_file in enumerate(uploaded_files, 1):
                try:
                    # Read image
                    image = PILImage.open(uploaded_file).convert('RGB')  # Use PILImage to avoid conflict
                    image_np = np.array(image)
                    st.subheader(f"Processing {uploaded_file.name}")
                    st.image(image_np, caption='Uploaded Image', use_column_width=True)

                    # YOLOv7 Inference
                    detections = run_yolo_inference(yolo_model, image_np, conf_thres=YOLO_CONF_THRESHOLD, 
                                                    iou_thres=YOLO_IOU_THRESHOLD, device=DEVICE)
                    st.write(f"Number of chessboards detected: {len(detections)}")

                    if not detections:
                        st.warning("No chessboards detected in this image.")
                        continue

                    # Draw Bounding Boxes
                    annotated_image = draw_yolo_boxes(image_np.copy(), detections, YOLO_CLASS_NAMES)
                    st.image(annotated_image, caption='Annotated Image', use_column_width=True)

                    # Process Each Detection
                    for det_idx, det in enumerate(detections, 1):
                        cls = int(det[5])
                        if YOLO_CLASS_NAMES[cls] != "chessboard":
                            continue  # Skip if not a chessboard

                        # Assign a unique key for each checkbox
                        checkbox_label = f"Show grids for Chessboard {det_idx} in {uploaded_file.name}"
                        checkbox_key = f"show_grids_{img_idx}_{det_idx}"
                        show_grids = st.checkbox(checkbox_label, key=checkbox_key)

                        if show_grids:
                            # Apply the corner detection and perspective transform pipeline
                            processed_img, warped = process_chessboard(image_np, det)
                            if warped is not None:
                                # Predict FEN
                                fen = predict_fen_from_image(fen_model, warped)
                                st.write(f"**Predicted FEN**: `{fen}`")

                                # Display Warped Chessboard
                                st.image(warped, caption=f'Warped Chessboard {det_idx} in {uploaded_file.name}', use_column_width=True)

                                # Visualize the grids for debugging
                                visualize_grids(warped, title=f"Chessboard {det_idx} Grids in {uploaded_file.name}")
                            else:
                                st.warning(f"Failed to process chessboard {det_idx} in {uploaded_file.name}.")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

    elif option == "Use Webcam":
        st.sidebar.write("Click the button below to start webcam and capture images.")
        webrtc_ctx = webrtc_streamer(
            key="chessboard-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: VideoProcessor(yolo_model, fen_model),
            async_processing=True,
        )

        # Placeholder for FEN results
        st.write("### FEN Results")
        if webrtc_ctx.video_processor:
            fen_list = webrtc_ctx.video_processor.fen_results if hasattr(webrtc_ctx.video_processor, 'fen_results') else []
            for idx, fen in enumerate(fen_list, 1):
                st.write(f"**Chessboard {idx} FEN**: `{fen}`")

# ---------------------------- Entry Point ----------------------------

if __name__ == "__main__":
    main()
