# Chess FEN Prediction Project

This project focuses on chessboard detection, FEN (Forsyth–Edwards Notation) prediction, and related tasks using models like YOLOv7, AlexNet, EfficientNet, and Vision Transformer. This README provides instructions for setting up the project, running the code, and utilizing pre-trained weights.

---

## Dataset Preparation

### 1. Chess FEN Prediction Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/koryakinp/chess-positions). The dataset includes:
- Train images and labels
- Test images and labels

### 2. YOLO Dataset
Download the YOLO dataset from [Roboflow](https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces/dataset/4). It is required for chessboard detection using YOLO.

---

## Pre-trained Weights

- Pre-trained weights for both YOLO and FEN models are available in this [Google Drive folder](https://drive.google.com/drive/folders/1-xuac4Z_l6Sc06tr6IIWzEoZNO8kvgtW).
- You can either use these weights or train the models from scratch.

---

## Instructions for Training and Using Models

### 1. YOLOv7 Training
To train YOLO from scratch, use the following command:

python3 yolo_train.py --workers 8 --device 0 --batch-size 32 --data data/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

After training, the best model will be stored. This model can be used in the Streamlit app for inference.

### 2. Running FEN Models
Train and validate the models (AlexNet, EfficientNet, Vision Transformer) for FEN prediction:
- Use the respective training scripts: `alexnet.py`, `efficientnet.py`, and `vision_transformer.py`.
- For EfficientNetB0, train using `EfficientNetB0_Model_Train_Script.py` and validate using `EfficientNetB0_Model_Test_Script.py`.
- Store the best-performing models in a folder named `FEN_models`. The models will be saved as `.pth` files.

---

## Streamlit Application

### Running the App
1. Ensure both YOLO and FEN models are trained or their weights are available and map them to the respective paths in the Streamlit code.
2. Execute the Streamlit application using the following command:

streamlit run streamlit.py

---

## File Descriptions

- `yolo_train.py`: Script to train YOLOv7 for chessboard detection.
- `alexnet.py`: Script for training and evaluating AlexNet on the FEN prediction task.
- `efficientnet.py`: Script for training EfficientNet for FEN prediction.
- `vision_transformer.py`: Script for Vision Transformer-based FEN prediction.
- `streamlit.py`: Streamlit app for chessboard detection and FEN prediction.
