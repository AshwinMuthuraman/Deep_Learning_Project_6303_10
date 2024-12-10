# Generating FEN-notations from Chess Board Images

## Project Overview
This project, undertaken by our Group, focuses on the development of a deep learning model to identify and classify positions on a chessboard. Our goal is to accurately translate these positions into FEN (Forsyth-Edwards Notation) using labeled image data.

## Team Members
- Sudhanshu Dotel
- Bhanu Sai Praneeth Sarva
- Ashwin Muthuraman
- Bhoomika Nanjaraja

## Objectives
The main objective of this project is to train a model that can precisely recognize different chessboard configurations and encode them into FEN notations. This involves detecting individual pieces and their respective positions on the board.

## Dataset
We are utilizing a dataset available on Kaggle, which can be accessed [here](https://www.kaggle.com/datasets/koryakinp/chess-positions/data). This dataset includes images of chessboards annotated with FEN notations, ideal for training our models.

## Methodology
To achieve our goals, we are experimenting with various models:
- **AlexNet**
- **EfficientNet**
- **Vision Transformers (ViT)**

These models are evaluated based on their accuracy in predicting the chessboard positions. Additionally, we implement **YOLO** (You Only Look Once) object detection for handling images with background noise and for real-time detection in live webcam feeds.

## Explainability
We use **Captum**, an open-source library for model interpretability, to understand how our models make decisions. This insight helps refine our approach and improve model performance.

## Frontend Interface
For a user-friendly experience, we are developing a frontend application using **Streamlit**. This interface will allow users to upload images of chessboards and view the corresponding FEN notation as predicted by our models.
NOTE : for accessing the model weights file for streamlit, please download the weights file from the following link : https://drive.google.com/drive/folders/1-xuac4Z_l6Sc06tr6IIWzEoZNO8kvgtW?usp=sharing 
## Project Significance
This project stands at the intersection of advanced machine learning techniques and game theory, offering valuable insights into automated game analysis and enhancing AI applications in strategy games like chess.
