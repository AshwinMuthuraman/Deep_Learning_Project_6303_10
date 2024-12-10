import streamlit as st
import cv2
import numpy as np
import torch
import torchvision  # For NMS and models
from PIL import Image as PILImage  # Aliased to prevent conflicts
import albumentations as A
from albumentations.pytorch import ToTensorV2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import os
from typing import List, Tuple, Dict

# Import matplotlib.pyplot and alias as plt
import matplotlib.pyplot as plt  # Ensure this import is present

# ---------------------------- Constants and Configurations ----------------------------

# YOLOv7 Configuration
YOLO_WEIGHTS_PATH = "/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/Deep_Learning_Project_6303_10/YOLO.pt"  # Path to YOLO model weights
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_CLASS_NAMES = ["chessboard"]  # List of class names

# FEN Models Configuration
FEN_MODELS_CONFIG = {
    "AlexNet": {
        "model_path": "/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/Deep_Learning_Project_6303_10/FEN_MODELS/ALEX_NET_best_model.pth",
        "architecture": "alexnet"
    },
    "Vision Transformer": {
        "model_path": "/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/Deep_Learning_Project_6303_10/FEN_MODELS/VIT_with_bw_images.pth",
        "architecture": "vit"
    },
    "EfficientNet-B7": {  # Updated EfficientNet-B7 configuration
        "model_path": "/home/praneeth/GWU/FALL_24/DEEP_LEARNING/Project/Deep_Learning_Project_6303_10/FEN_MODELS/effnet_best_model.pth",
        "architecture": "efficientnet-b7"  # Changed from 'efficientnet_b7' to 'efficientnet-b7'
    }
}

# FEN Model Specifics
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
def load_yolo_model(weights_path: str, device: torch.device):
    """
    Loads the YOLOv7 model from the provided weights file.
    """
    try:
        # Assuming YOLOv7 model is saved as a TorchScript or similar format
        # Adjust the loading method based on how the model was saved
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
    y = x.clone()
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

def run_yolo_inference(model, image: np.ndarray, conf_thres=0.25, iou_thres=0.45, device='cuda'):
    """
    Perform inference on an image using YOLOv7.
    """
    original_shape = image.shape[:2]  # Height, Width

    # Preprocess Image
    img, ratio, pad = letterbox(image, new_shape=(640, 640))
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
def load_fen_models(config: Dict[str, Dict[str, str]], device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Load all FEN models based on the provided configuration.
    Returns a dictionary mapping model names to model instances.
    """
    models = {}
    for model_name, model_info in config.items():
        model_path = model_info["model_path"]
        architecture = model_info["architecture"]
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            continue
        try:
            if architecture.lower() == "alexnet":
                model = define_alexnet_model(num_classes=13)
            elif architecture.lower() == "vit":
                model = define_vit_model(num_classes=13)
                if model is None:
                    st.error(f"Failed to define ViT model for '{model_name}'.")
                    continue
            elif architecture.lower() == "efficientnet-b7":
                model = define_efficientnet_b7_model(num_classes=13)
                if model is None:
                    st.error(f"Failed to define EfficientNet-B7 model for '{model_name}'.")
                    continue
            else:
                st.error(f"Unsupported architecture '{architecture}' for model '{model_name}'.")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models[model_name] = model
            st.success(f"Loaded FEN model: {model_name}")
        except Exception as e:
            st.error(f"Error loading model '{model_name}': {e}")
    return models

def define_alexnet_model(num_classes=13):
    """
    Define the AlexNet architecture for FEN prediction.
    """
    model = torchvision.models.alexnet(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier to match the number of classes
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(256 * 6 * 6, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(4096, num_classes),
    )
    return model.to(DEVICE)

def define_vit_model(num_classes=13):
    """
    Define the Vision Transformer (ViT) architecture for FEN prediction.
    Ensures that the classifier head matches the training configuration.
    """
    try:
        from torchvision.models import vit_b_16

        # Initialize ViT-B/16 model without pretrained weights
        model = vit_b_16(pretrained=False)

        # Freeze all layers except the classifier head
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier head to match the number of classes
        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, num_classes)

        return model.to(DEVICE)
    except ImportError:
        st.error("Vision Transformer (ViT) is not available in your torchvision version.")
        return None

def define_efficientnet_b7_model(num_classes=13):
    """
    Define the EfficientNet-B7 architecture for FEN prediction.
    """
    try:
        from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

        # Initialize EfficientNet-B7 model
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if torch.cuda.is_available() else None
        model = efficientnet_b7(weights=weights)

        # Freeze all layers except the classifier head
        for param in model.parameters():
            param.requires_grad = False

        # Replace the classifier head
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(num_ftrs, num_classes)
        )

        return model.to(DEVICE)
    except ImportError:
        st.error("EfficientNet-B7 is not available in your torchvision version.")
        return None

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
            if grid.shape[0] == grid_h and grid.shape[1] == grid_w:
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

def predict_fen_from_image(models: Dict[str, torch.nn.Module], image, use_yolo=False):
    """
    Predicts the FEN notation for a given chessboard image using selected models.
    If use_yolo is True, the image is assumed to be a warped chessboard already.
    If use_yolo is False, the image is treated as a single chessboard without detection.
    Returns a dictionary mapping model names to their predicted FEN strings.
    """
    if image is None:
        st.error("No image provided for FEN prediction.")
        return {}

    if use_yolo:
        # The image is already a warped chessboard (cropped and transformed)
        chessboards = [image]
    else:
        # The image is treated as a single chessboard
        chessboards = [image]

    fen_predictions = {}
    for model_name, model in models.items():
        predicted_labels = []
        for chessboard in chessboards:
            try:
                # Convert to grayscale if not already
                if len(chessboard.shape) == 3 and chessboard.shape[2] == 3:
                    chessboard_gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
                else:
                    chessboard_gray = chessboard.copy()

                # Split image into grids
                grids = split_image_into_grids(chessboard_gray)
                if len(grids) != 64:
                    st.warning(f"Chessboard image does not have 64 grids (got {len(grids)}). Check cropping and warping.")
                    continue

                for grid_idx, grid in enumerate(grids, 1):
                    # Preprocess grid
                    grid_resized = cv2.resize(grid, (FEN_IMAGE_SIZE, FEN_IMAGE_SIZE))
                    if len(grid_resized.shape) == 2:  # Grayscale to RGB
                        grid_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2RGB)
                    else:
                        grid_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(image=np.array(grid_rgb))['image'].unsqueeze(0).to(DEVICE)

                    # Model prediction
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, pred = torch.max(outputs, 1)
                        predicted_labels.append(pred.item())
            except Exception as e:
                st.warning(f"Error processing chessboard with model {model_name}: {e}")
                predicted_labels.append(0)  # Assuming '0' is 'empty'

        # Convert labels to FEN
        fen = label_to_fen(predicted_labels)
        fen_predictions[model_name] = fen

    return fen_predictions

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
    def __init__(self, yolo_model, fen_models):
        self.yolo_model = yolo_model
        self.fen_models = fen_models
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
                # Predict FEN with selected models
                fen_predictions = predict_fen_from_image(self.fen_models, warped, use_yolo=True)
                fen_text = "; ".join([f"{model}: {fen}" for model, fen in fen_predictions.items()])
                self.fen_results.append(fen_text)

                # Overlay FEN annotation near the bounding box on the annotated image
                x1, y1, x2, y2 = map(int, det[:4])
                # Position the text slightly above the top-left corner of the bounding box
                text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                cv2.putText(annotated_image, fen_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Limit the number of stored FEN results to the last 10
                if len(self.fen_results) > 10:
                    self.fen_results.pop(0)

        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# ---------------------------- Streamlit Application ----------------------------

def main():
    st.title("Chessboard Detection and FEN Prediction")
    st.write("Choose between using YOLO for chessboard detection with webcam/image upload or directly predict FEN from uploaded images.")

    # Mode Selection
    mode = st.radio(
        "Select Mode:",
        ("Use YOLO for Chessboard Detection", "Direct FEN Prediction")
    )

    # Load FEN Models (always needed)
    fen_models = load_fen_models(FEN_MODELS_CONFIG, DEVICE)

    if mode == "Use YOLO for Chessboard Detection":
        # Load YOLO model
        yolo_model = load_yolo_model(YOLO_WEIGHTS_PATH, DEVICE)

        if yolo_model is None or not fen_models:
            st.error("Failed to load the necessary models. Please check the model paths.")
            return

        # Input Method Selection
        input_method = st.selectbox(
            "Choose Input Method:",
            ("Webcam", "Upload Image")
        )

        if input_method == "Webcam":
            # Webcam Input with YOLO
            st.write("### Webcam Chessboard Detection and FEN Prediction")
            webrtc_ctx = webrtc_streamer(
                key="chessboard-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: VideoProcessor(yolo_model, fen_models),
                async_processing=True,
            )

            # Placeholder for FEN results
            st.write("### FEN Results")
            if webrtc_ctx.video_processor:
                fen_list = webrtc_ctx.video_processor.fen_results if hasattr(webrtc_ctx.video_processor, 'fen_results') else []
                for idx, fen in enumerate(fen_list, 1):
                    st.write(f"**Chessboard {idx} FEN**: `{fen}`")

        elif input_method == "Upload Image":
            # Image Upload with YOLO
            st.write("### Image Upload Chessboard Detection and FEN Prediction")
            uploaded_files = st.file_uploader("Upload Image(s)", type=IMAGE_EXTENSIONS, accept_multiple_files=True)
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
                                    # Predict FEN with selected models
                                    fen_predictions = predict_fen_from_image(fen_models, warped, use_yolo=True)
                                    for model_name, fen in fen_predictions.items():
                                        st.write(f"**{model_name} Predicted FEN**: `{fen}`")

                                    # Display Warped Chessboard
                                    st.image(warped, caption=f'Warped Chessboard {det_idx} in {uploaded_file.name}', use_column_width=True)

                                    # Visualize the grids for debugging
                                    visualize_grids(warped, title=f"Chessboard {det_idx} Grids in {uploaded_file.name}")
                                else:
                                    st.warning(f"Failed to process chessboard {det_idx} in {uploaded_file.name}.")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

    elif mode == "Direct FEN Prediction":
        # Direct FEN Prediction without YOLO
        st.write("### Direct FEN Prediction from Uploaded Images")
        uploaded_files = st.file_uploader("Upload Image(s)", type=IMAGE_EXTENSIONS, accept_multiple_files=True)
        if uploaded_files:
            for img_idx, uploaded_file in enumerate(uploaded_files, 1):
                try:
                    # Read image
                    image = PILImage.open(uploaded_file).convert('RGB')  # Use PILImage to avoid conflict
                    image_np = np.array(image)
                    st.subheader(f"Processing {uploaded_file.name}")
                    st.image(image_np, caption='Uploaded Image', use_column_width=True)

                    # Assign a unique key for each checkbox
                    checkbox_label = f"Show grids for Chessboard {img_idx} in {uploaded_file.name}"
                    checkbox_key = f"show_grids_direct_{img_idx}"
                    show_grids = st.checkbox(checkbox_label, key=checkbox_key)

                    if show_grids:
                        # Split the image into grids directly
                        # Convert to grayscale if not already
                        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                            image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                        else:
                            image_gray = image_np.copy()

                        # Split image into grids
                        grids = split_image_into_grids(image_gray)
                        if len(grids) != 64:
                            st.warning(f"Uploaded image does not have 64 grids (got {len(grids)}). Check image dimensions.")
                            continue

                        # Predict FEN with selected models
                        fen_predictions = predict_fen_from_image(fen_models, image_np, use_yolo=False)
                        for model_name, fen in fen_predictions.items():
                            st.write(f"**{model_name} Predicted FEN**: `{fen}`")

                        # Visualize the grids for debugging
                        visualize_grids(image_np, title=f"Chessboard Grids in {uploaded_file.name}")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

if __name__ == "__main__":
    main()
