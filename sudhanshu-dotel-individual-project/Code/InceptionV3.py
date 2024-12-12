# -*- coding: utf-8 -*-
"""
Advanced Chess FEN Generation using EfficientNet-B7
Leveraging NVIDIA A10G GPU for optimized training
"""
import os
import glob
import random
import datetime
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm  # For progress bars



# ===============================
# Configuration and Hyperparameters
# ===============================

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# Configuration Class
class Config:
    # Paths
    ORIGINAL_PATH = '/home/ubuntu/FinalProject/'
    DATA_DIR = os.path.join(ORIGINAL_PATH, 'dataset')  # Assuming dataset folder is in current directory
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    # Training parameters
    N_EPOCHS = 5
    BATCH_SIZE = 32  #best for inception v3
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 299
    CHANNELS = 3
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SAVE_MODEL = True
    EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before early stopping

    # DataLoader parameters
    NUM_WORKERS = 8  # Adjust based on CPU cores
    PIN_MEMORY = True  # For faster data transfer to CUDA

    # Model parameters
    OUTPUTS_A = 13  # Number of classes

    # Labels
    FEN_TO_LABEL_DICT = {
        'p': 1, 'P': 2,
        'b': 3, 'B': 4,
        'r': 5, 'R': 6,
        'n': 7, 'N': 8,
        'q': 9, 'Q': 10,
        'k': 11, 'K': 12,
        '0': 0  # Representing empty grid
    }

    LABEL_TO_FEN_SHORT = {
        0: "emptyGrid",
        1: "p", 2: "P",
        3: "b", 4: "B",
        5: "r", 6: "R",
        7: "n", 8: "N",
        9: "q", 10: "Q",
        11: "k", 12: "K"
    }

    # Model Save Paths
    TIMESTAMP = datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
    NICKNAME = "EfficientNetB7_Model"
    MODEL_SAVE_PATH = os.path.join('Output_Data', f"{NICKNAME}_{TIMESTAMP}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    MODEL_SAVE_NAME = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
    PLOTS_SAVE_PATH = MODEL_SAVE_PATH
    PREDICTION_EXCEL = 'test_predictions.xlsx'  # Name of the prediction Excel file


# Initialize configuration
cfg = Config()


# ===============================
# Utility Functions
# ===============================

def fen_to_label(fen: str) -> List[int]:
    """
    Convert a FEN string to a list of labels corresponding to each grid.
    """
    labels = []
    rows = fen.split('-')
    for row in rows:
        for char in row:
            if char.isalpha():
                labels.append(cfg.FEN_TO_LABEL_DICT.get(char, 0))
            elif char.isdigit():
                labels.extend([0] * int(char))
    return labels


def label_to_fen(label_list: List[int]) -> str:
    """
    Convert a list of labels back to a FEN string.
    """
    fen_notation = ''
    empty_count = 0
    for idx, label in enumerate(label_list):
        if label == 0:
            empty_count += 1
        else:
            if empty_count > 0:
                fen_notation += str(empty_count)
                empty_count = 0
            fen_notation += cfg.LABEL_TO_FEN_SHORT.get(label, '0')
        if (idx + 1) % 8 == 0:
            if empty_count > 0:
                fen_notation += str(empty_count)
                empty_count = 0
            if idx != 63:
                fen_notation += '-'
    return fen_notation


def split_image_into_grids(image: np.ndarray, grid_size: Tuple[int, int] = (50, 50)) -> List[np.ndarray]:
    """
    Split the chessboard image into 64 grids.
    """
    grids = []
    h, w = image.shape
    grid_h, grid_w = grid_size
    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            grid = image[i:i + grid_h, j:j + grid_w]
            if grid.shape[0] == grid_h and grid.shape[1] == grid_w:
                grids.append(grid)
    return grids


# ===============================
# Dataset Class
# ===============================

class ChessDataset(Dataset):
    """
    Custom Dataset for Chess Piece Classification.
    Processes images on-the-fly without pre-processing.
    """

    def __init__(self, folder: str, transform=None):
        """
        Args:
            folder (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = self.get_image_paths(folder)
        self.transform = transform
        self.labels = self.prepare_labels()

    def get_image_paths(self, folder: str) -> List[str]:
        """
        Retrieve image file paths from the specified folder.
        """
        extensions = ['.jpeg', '.jpg', '.JPEG', '.JPG', '.png', '.bmp', '.gif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        return image_paths

    def prepare_labels(self) -> List[List[int]]:
        """
        Extract labels from image filenames based on FEN.
        """
        labels = []
        for path in self.image_paths:
            filename = os.path.splitext(os.path.basename(path))[0]
            labels.append(fen_to_label(filename))
        return labels

    def __len__(self):
        return len(self.image_paths) * 64  # 64 grids per image

    def __getitem__(self, idx):
        image_idx = idx // 64
        grid_idx = idx % 64
        img_path = self.image_paths[image_idx]
        label = self.labels[image_idx][grid_idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing images by using a default image
            print(f"Warning: Image {img_path} could not be read. Using blank grid.")
            image_rgb = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS), dtype=np.uint8)
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grids = split_image_into_grids(image_gray, grid_size=(50, 50))
            if grid_idx >= len(grids):
                # Handle cases where grid_idx is out of bounds
                print(f"Warning: Grid index {grid_idx} out of bounds for image: {img_path}. Using blank grid.")
                image_rgb = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS), dtype=np.uint8)
            else:
                grid = grids[grid_idx]
                grid_resized = cv2.resize(grid, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
                image_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(image_rgb)

        if self.transform:
            image = self.transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label



train_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),  # Resize images to the required input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet statistics
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),  # Resize images to the required input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet statistics
                         std=[0.229, 0.224, 0.225])
])


# ===============================
# Data Preparation
# ===============================

def prepare_dataloaders(folder: str, transform, batch_size: int = cfg.BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """
    Prepare DataLoader for a given dataset.

    Args:
        folder (str): Directory with all the images.
        transform (callable): Transformations to apply.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = ChessDataset(folder=folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
                            pin_memory=cfg.PIN_MEMORY)
    return dataloader


# ===============================
# Model Definition with EfficientNet-B7
# ===============================

# def get_model(pretrained: bool = True, num_classes: int = cfg.OUTPUTS_A) -> nn.Module:
#     """
#     Initialize and return the pre-trained EfficientNet-B7 model.
#     """
#     model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None)
#
#     # Freeze all layers except the classifier
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Replace the classifier head
#     num_ftrs = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.5, inplace=True),
#         nn.Linear(num_ftrs, num_classes)
#     )
#
#     return model.to(cfg.DEVICE)

def get_model(pretrained: bool = True, num_classes: int = cfg.OUTPUTS_A) -> nn.Module:
    """
    Initialize and return the pre-trained Inception v3 model with partial layer freezing.
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 2 inception blocks and the classifier
    # Inception v3 has layers named as follows for the last 2 blocks:
    # - Mixed_7b
    # - Mixed_7c
    for name, param in model.named_parameters():
        if "Mixed_7b" in name or "Mixed_7c" in name or "fc" in name:
            param.requires_grad = True

    # Replace the classifier
    model.aux_logits = False  # Disable auxiliary outputs for simplicity
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace with the correct number of classes

    return model.to(cfg.DEVICE)




# ===============================
# Early Stopping Class
# ===============================

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


# ===============================
# Training and Validation Function
# ===============================

def train_and_validate(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs: int = cfg.N_EPOCHS,
        patience: int = cfg.EARLY_STOPPING_PATIENCE,
        save_path: str = cfg.MODEL_SAVE_NAME
) -> dict:
    """
    Train and validate the model.
    """
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    best_val_loss = float('inf')

    # Lists to store metrics
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = images.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train.double() / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                images = images.to(cfg.DEVICE, non_blocking=True)
                labels = labels.to(cfg.DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_val.double() / len(val_loader.dataset)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy.item())
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy.item())

        # Print metrics
        epoch_time = (time.time() - start_time) / 60  # in minutes
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f} "
            f"| Time: {epoch_time:.2f} min"
        )

        # Early Stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if cfg.SAVE_MODEL:
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with val_loss: {best_val_loss:.4f}")

    print('Training complete')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    # Save training and validation metrics plots
    os.makedirs(cfg.PLOTS_SAVE_PATH, exist_ok=True)
    plot_metrics(history, cfg.PLOTS_SAVE_PATH)
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return history


# ===============================
# Plotting Function
# ===============================

def plot_metrics(history: dict, save_path: str):
    """
    Plot training and validation metrics.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(dpi=300)
    plt.plot(epochs, history['train_loss'], '-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], '-o', label='Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_path, "train_valid_loss.png"))
    plt.close()
    print(f"Saved loss plot to {os.path.join(save_path, 'train_valid_loss.png')}")

    # Plot Accuracy
    plt.figure(dpi=300)
    plt.plot(epochs, history['train_accuracy'], '-o', label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], '-o', label='Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(save_path, "train_valid_accuracy.png"))
    plt.close()
    print(f"Saved accuracy plot to {os.path.join(save_path, 'train_valid_accuracy.png')}")


# ===============================
# Evaluation Function
# ===============================
def evaluate_model(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test set and return metrics.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct.double() / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    print(f'Accuracy of the network on the {total} test grids: {accuracy * 100:.2f} %')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return accuracy.item(), precision, recall, f1


# ===============================
# Prediction Function
# ===============================

def make_prediction(model: nn.Module, device: str, image_path: str, transform=None) -> str:
    """
    Predict the FEN notation for a given chessboard image.
    """
    model.eval()
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Image {image_path} could not be read. Using blank grid.")
        image_rgb = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS), dtype=np.uint8)
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grids = split_image_into_grids(image_gray, grid_size=(50, 50))
        predicted_labels = []

        if transform is None:
            transform = A.Compose([
                A.Resize(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE),  # Ensure resizing
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        for grid in grids:
            grid_resized = cv2.resize(grid, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            grid_rgb = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(grid_rgb)
            input_tensor = transform(image=np.array(pil_image))['image']
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_labels.append(pred.item())

        fen_notation = label_to_fen(predicted_labels)
        return fen_notation


def create_excel_file(image_paths: List[str], labels: List[int], excel_path: str):
    """
    Create an Excel file with image paths, FEN notations, and full labels corresponding to FEN.
    Each row corresponds to a single image.
    """
    data = []
    unique_images = list(set(image_paths))  # Ensure unique images
    for img_path in unique_images:
        fileName = os.path.splitext(os.path.basename(img_path))[0]
        fen = fileName
        # Extract labels corresponding to this image
        image_labels = [label for path, label in zip(image_paths, labels) if path == img_path]
        # Convert list of labels to a comma-separated string
        labels_str = ','.join(map(str, image_labels))
        data.append([img_path, fen, labels_str])

    df = pd.DataFrame(data, columns=['image_path', 'fen', 'labels'])
    df.to_excel(excel_path, index=False)
    print(f"Excel file created at {excel_path}")


# ===============================
# Main Function
# ===============================



def main():
    """
    Main function to orchestrate data loading, training, validation, and prediction.
    """
    os.chdir(cfg.ORIGINAL_PATH)

    # Prepare DataLoaders
    print("Starting data preparation...")

    train_loader = prepare_dataloaders(
        folder=cfg.TRAIN_DIR,
        transform=train_transform,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )

    val_loader = prepare_dataloaders(
        folder=cfg.TRAIN_DIR,
        transform=val_transform,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )

    test_loader = prepare_dataloaders(
        folder=cfg.TEST_DIR,
        transform=val_transform,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )

    print(f"Number of training grids: {len(train_loader.dataset)}")
    print(f"Number of validation grids: {len(val_loader.dataset)}")

    # Initialize the model
    print("Initializing EfficientNet-B7 model...")
    model = get_model(pretrained=True, num_classes=cfg.OUTPUTS_A)
    print("Model initialized successfully.\n")

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train and validate
    print("Starting training and validation...")
    history = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.N_EPOCHS,
        patience=cfg.EARLY_STOPPING_PATIENCE,
        save_path=cfg.MODEL_SAVE_NAME
    )

    # Plot metrics
    print("Plotting metrics...")
    plot_metrics(history, cfg.PLOTS_SAVE_PATH)

    # Evaluate on the test set
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_NAME))
    test_accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



if __name__ == '__main__':
    main()