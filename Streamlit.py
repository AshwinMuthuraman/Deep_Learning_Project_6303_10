import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_TO_FEN_SHORT = {
    0: "0", 1: "p", 2: "P",
    3: "b", 4: "B",
    5: "r", 6: "R",
    7: "n", 8: "N",
    9: "q", 10: "Q",
    11: "k", 12: "K"
}

# Define your model architecture
def define_model(num_classes=13):
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)  # Do not load pretrained weights
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_ftrs, num_classes)  # Match your training setup
    return model

# Load the trained model weights
@st.cache_resource
def load_model(model_path):
    model = define_model(num_classes=13)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Splitting the image into grids
def split_image_into_grids(image):
    h, w = image.shape
    grid_h, grid_w = h // 8, w // 8
    grids = []
    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            grid = image[i:i + grid_h, j:j + grid_w]
            grids.append(grid)
    return grids

# Convert labels to FEN notation
def label_to_fen(label_list):
    fen = ""
    empty_count = 0
    for idx, label in enumerate(label_list):
        if label == 0:
            empty_count += 1
        else:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            fen += LABEL_TO_FEN_SHORT.get(label, "0")
        if (idx + 1) % 8 == 0:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            if idx != len(label_list) - 1:
                fen += "/"
    return fen

# Preprocessing the image
transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Predict FEN for a chessboard image
def predict_fen(model, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    grids = split_image_into_grids(image_gray)
    if len(grids) != 64:
        raise ValueError(f"Image does not have 64 grids (got {len(grids)}). Check cropping.")

    predicted_labels = []
    for grid in grids:
        grid_resized = cv2.resize(grid, (IMAGE_SIZE, IMAGE_SIZE))
        grid_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2RGB)
        input_tensor = transform(image=np.array(grid_rgb))["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_labels.append(pred.item())

    return label_to_fen(predicted_labels)

# Streamlit Application
st.title("Chess FEN Prediction App")
st.write("Upload a chessboard image to predict the FEN representation.")

# Upload the model file
model_path = "/Users/bhoomikan/Documents/Deep_Learning/Project/VIT_early_stop_best_model.pth"
if not model_path:
    st.warning("Please provide the path to your trained model.")

# Load the model
if model_path:
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Image upload
uploaded_image = st.file_uploader("Upload a chessboard image (JPEG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Predict FEN
    if st.button("Predict FEN"):
        try:
            predicted_fen = predict_fen(model, image)
            st.subheader("Predicted FEN Notation:")
            st.text(predicted_fen)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
