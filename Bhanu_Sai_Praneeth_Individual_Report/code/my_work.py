# -*- coding: utf-8 -*-
"""
Advanced Chess FEN Generation using Vision Transformer (ViT)
Leveraging NVIDIA A10G GPU for optimized training with Explainability using Captum
"""

import os
import glob
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data, tensorboard
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import Captum libraries for explainability
from captum.attr import IntegratedGradients
import torch.nn.functional as F

# ===============================
# Suppress Future Warnings (Optional)
# ===============================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================
# Configuration and Hyperparameters
# ===============================

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Paths
ORIGINAL_PATH = os.getcwd()
DATASET_PATH = '/home/ubuntu/project_DL/chess-positions/versions/1/dataset'
TRAIN_FOLDER = os.path.join(DATASET_PATH, 'train')
TEST_FOLDER = os.path.join(DATASET_PATH, 'test')
MODEL_SAVE_PATH = 'Output_Data_test_bw'
LOG_DIR = os.path.join(MODEL_SAVE_PATH, 'logs_test_bw')
EXPLAINABILITY_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'explanations_test_bw')

# Hyperparameters
NUM_EPOCHS = 10  # Adjust as needed
BATCH_SIZE = 256  # Adjust based on GPU memory
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
LABEL_SIZE = 13
SEED = 42
NUM_WORKERS = 8  # Adjust based on CPU cores

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===============================
# Label Definitions
# ===============================

# Mapping from FEN characters to labels
FEN_TO_LABEL = {
    'p': 1, 'P': 2,
    'b': 3, 'B': 4,
    'r': 5, 'R': 6,
    'n': 7, 'N': 8,
    'q': 9, 'Q': 10,
    'k': 11, 'K': 12,
    '0': 0  # Empty grid
}

LABEL_TO_FEN_SHORT = {
    0: '0',
    1: 'p', 2: 'P',
    3: 'b', 4: 'B',
    5: 'r', 6: 'R',
    7: 'n', 8: 'N',
    9: 'q', 10: 'Q',
    11: 'k', 12: 'K'
}

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
                labels.append(FEN_TO_LABEL.get(char, 0))
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
            fen_notation += LABEL_TO_FEN_SHORT.get(label, '0')
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
    if len(image.shape) == 2:
        # If image is grayscale, add channel dimension
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w, c = image.shape
    grid_h, grid_w = grid_size
    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            grid = image[i:i + grid_h, j:j + grid_w]
            grids.append(grid)
    return grids

# ===============================
# Dataset Class
# ===============================

class ChessDataset(data.Dataset):
    def __init__(self, image_paths: List[str], transform=None, augmentation_prob: float = 0.5):
        """
        Args:
            image_paths (List[str]): List of image file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            augmentation_prob (float): Probability of converting an image to grayscale.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.labels = self.prepare_labels()
        self.augmentation_prob = augmentation_prob  # Probability to apply grayscale

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
        try:
            image_idx = idx // 64
            grid_idx = idx % 64
            img_path = self.image_paths[image_idx]
            label = self.labels[image_idx][grid_idx]

            # Load image in color (BGR)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to read image: {img_path}")

            grids = split_image_into_grids(image, grid_size=(50, 50))
            if grid_idx >= len(grids):
                raise IndexError(f"Grid index {grid_idx} out of range for image: {img_path}")
            grid = grids[grid_idx]

            # Randomly convert to grayscale based on augmentation probability
            if random.random() < self.augmentation_prob:
                grid_gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
                grid_rgb = cv2.cvtColor(grid_gray, cv2.COLOR_GRAY2RGB)
            else:
                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

            # Resize the grid
            grid_resized = cv2.resize(grid_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            pil_image = Image.fromarray(grid_resized)

            if self.transform:
                # Albumentations expects images in numpy array format
                transformed = self.transform(image=np.array(pil_image))
                image = transformed['image']
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                image = transform(pil_image)

            return image, label
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            # Return a default tensor and label or handle as appropriate
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), 0

# ===============================
# Data Augmentation and Transforms
# ===============================

# Advanced data augmentation using Albumentations
train_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),  # Ensure resizing
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(p=0.3),
    A.CLAHE(p=0.5),
    # Optional: Additional augmentations
    A.RandomGamma(p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),  # Ensure resizing
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ===============================
# Data Preparation
# ===============================

def prepare_dataloaders(train_size: int = None, test_size: int = None, batch_size: int = BATCH_SIZE) -> Tuple[
    data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Prepare training, validation, and testing dataloaders.

    Args:
        train_size (int): Number of training images. If None, use all available.
        test_size (int or None): Number of test images. If None, use all available.
        batch_size (int): Batch size.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """

    def get_image_paths(folder: str) -> List[str]:
        # Supported image extensions
        extensions = ['.jpeg', '.jpg', '.JPEG', '.JPG', '.png', '.bmp', '.gif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        return image_paths

    # Get training image paths
    train_image_paths = get_image_paths(TRAIN_FOLDER)
    available_train = len(train_image_paths)
    print(f"Total training images found: {available_train}\n")

    if train_size is not None:
        if available_train < train_size:
            if available_train < 1:
                raise ValueError(
                    f"No training images found in {TRAIN_FOLDER}. Please check the directory and image formats.")
            print(
                f"Requested train_size={train_size} exceeds available images={available_train}. Using all available images.\n")
            train_size = available_train

        train_image_paths = random.sample(train_image_paths, train_size)

    # Shuffle before splitting to ensure randomness
    random.shuffle(train_image_paths)

    # Split training data into training and validation sets (80-20 split)
    split = int(0.8 * len(train_image_paths))
    train_paths, val_paths = train_image_paths[:split], train_image_paths[split:]

    # Create datasets
    train_dataset = ChessDataset(train_paths, transform=train_transform)
    val_dataset = ChessDataset(val_paths, transform=val_transform)

    # Get testing image paths
    test_image_paths = get_image_paths(TEST_FOLDER)
    available_test = len(test_image_paths)
    print(f"Total test images found: {available_test}\n")

    if test_size is not None:
        if available_test < test_size:
            if available_test < 1:
                raise ValueError(f"No test images found in {TEST_FOLDER}. Please check the directory and image formats.")
            print(
                f"Requested test_size={test_size} exceeds available images={available_test}. Using all available images.\n")
            test_size = available_test

        test_image_paths = random.sample(test_image_paths, test_size)

    # Shuffle test images as well
    random.shuffle(test_image_paths)

    # Create test dataset
    test_dataset = ChessDataset(test_image_paths, transform=val_transform)

    # Create DataLoaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                   pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True)

    # Logging the number of samples
    print(f"Number of training samples: {len(train_loader.dataset)}")  # Should be 80,000 * 64 = 5,120,000
    print(f"Number of validation samples: {len(val_loader.dataset)}")  # Should be 20% of training images * 64
    print(f"Number of test samples: {len(test_loader.dataset)}")        # Should be 20,000 * 64 = 1,280,000
    print()

    # Confirm batch counts
    print(f"Number of training batches per epoch: {len(train_loader)}")  # Should be 5,120,000 / 256 ≈ 20,000
    print(f"Number of validation batches per epoch: {len(val_loader)}")  # Should be 1,024,000 / 256 ≈ 4,000
    print(f"Number of test batches: {len(test_loader)}")                 # Should be 1,280,000 / 256 = 5,000
    print()

    return train_loader, val_loader, test_loader

# ===============================
# Model Definition with Vision Transformer
# ===============================

def get_model(pretrained: bool = True, num_classes: int = LABEL_SIZE) -> nn.Module:
    """
    Initialize and return the pre-trained Vision Transformer (ViT) model.
    """
    # Import Vision Transformer from torchvision
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)

   
    for param in model.parameters():
        param.requires_grad = False

    
    num_ftrs = model.heads.head.in_features  
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    return model.to(DEVICE)

# ===============================
# Early Stopping Class
# ===============================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
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
# Training and Evaluation Functions
# ===============================

def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                dataloaders: dict,
                num_epochs: int = NUM_EPOCHS,
                log_dir: str = LOG_DIR,
                patience: int = 7) -> nn.Module:
    """
    Train the model with AMP and Early Stopping, then return the best model based on validation loss.
    Also, collect training and validation metrics for plotting.
    """
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    # TensorBoard writer
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=os.path.join(MODEL_SAVE_PATH, 'early_stop_best_model.pth'))

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            samples_processed = 0  # Initialize counter
            batch_count = 0

            total_batches = len(dataloaders[phase])

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Iteration'):
                batch_count += 1

                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward with autocast
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                # backward with scaler
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                samples_processed += inputs.size(0)

                # Optional: Log progress every N batches
                if batch_count % 1000 == 0:
                    print(f'{phase.capitalize()} Epoch {epoch + 1}: Processed {batch_count}/{total_batches} batches.')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase.capitalize()} Epoch {epoch + 1} processed {samples_processed} samples.\n')

            # Log to TensorBoard
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)

            # Append metrics for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val':
                early_stopping(epoch_loss, model)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        if early_stopping.early_stop:
            break

    print('Training complete')
    print(f'Best Validation Loss: {best_loss:.4f}')

    # Close TensorBoard writer
    writer.close()

    # Save training and validation metrics plots
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # Load the best model weights saved by EarlyStopping
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'early_stop_best_model.pth'),
                                     map_location=DEVICE))
    return model

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot and save training vs validation loss and accuracy.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(MODEL_SAVE_PATH, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(MODEL_SAVE_PATH, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Saved accuracy plot to {acc_plot_path}")

def evaluate_model(model: nn.Module, dataloader: data.DataLoader) -> Tuple[float, float, float, float]:
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
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct.double() / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Grid-Level Accuracy of the network on the {total} test grids: {accuracy * 100:.2f} %')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return accuracy.item(), precision, recall, f1

def evaluate_model_image_level(model: nn.Module, dataloader: data.DataLoader, num_grids: int = 64) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test set at the image level by aggregating grid predictions.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (data.DataLoader): DataLoader for the test dataset.
        num_grids (int): Number of grids per image (default is 64 for an 8x8 chessboard).

    Returns:
        Tuple containing image-level Accuracy, Precision, Recall, and F1-Score.
    """
    model.eval()
    image_predictions = {}
    image_ground_truth = {}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc='Image-Level Evaluation')):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                grid_idx = batch_idx * dataloader.batch_size + i
                image_idx = grid_idx // num_grids  # Correctly map grid index to image index

                # Debugging Statements
                if grid_idx % 1000 == 0:
                    print(f"Processing Grid Index: {grid_idx}, Image Index: {image_idx}")

                if image_idx >= len(dataloader.dataset.image_paths):
                    print(f"Skipping grid_idx {grid_idx} as it exceeds the number of images.")
                    continue  # Skip if image_idx is out of range

                image_path = dataloader.dataset.image_paths[image_idx]
                if image_idx not in image_predictions:
                    image_predictions[image_idx] = []
                    # Extract ground truth FEN from filename
                    ground_truth_fen = os.path.splitext(os.path.basename(image_path))[0]
                    image_ground_truth[image_idx] = ground_truth_fen
                image_predictions[image_idx].append(preds[i].item())

    # Aggregate predictions for each image
    aggregated_predictions = []
    aggregated_ground_truth = []

    for image_idx in image_predictions:
        # Aggregation Strategy: Majority Voting or any other suitable method
        pred_labels = image_predictions[image_idx]
        pred_fen = label_to_fen(pred_labels)
        ground_truth_fen = image_ground_truth[image_idx]

        aggregated_predictions.append(pred_fen)
        aggregated_ground_truth.append(ground_truth_fen)

    # Calculate Metrics
    correct = sum([1 for pred, gt in zip(aggregated_predictions, aggregated_ground_truth) if pred == gt])
    total = len(aggregated_ground_truth)
    accuracy = correct / total if total > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        aggregated_ground_truth, aggregated_predictions, average='weighted', zero_division=0)

    print(f'Image-Level Accuracy of the network on the {total} test images: {accuracy * 100:.2f} %')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return accuracy, precision, recall, f1

def predict_fen(model: nn.Module, image_path: str, transform=None) -> str:
    """
    Predict the FEN notation for a given chessboard image.
    """
    model.eval()
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    grids = split_image_into_grids(image, grid_size=(50, 50))
    predicted_labels = []

    if transform is None:
        transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),  # Ensure resizing
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    for grid in grids:
        # Randomly decide to convert to grayscale based on the same probability used during training
        if random.random() < 0.5:
            grid_gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
            grid_rgb = cv2.cvtColor(grid_gray, cv2.COLOR_GRAY2RGB)
        else:
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        grid_resized = cv2.resize(grid_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        pil_image = Image.fromarray(grid_resized)
        input_tensor = transform(image=np.array(pil_image))['image']
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_labels.append(pred.item())

    fen_notation = label_to_fen(predicted_labels)
    return fen_notation

# ===============================
# Prediction Saving Function
# ===============================

def save_all_test_predictions(model: nn.Module, dataloader: data.DataLoader, csv_path: str = 'all_test_predictions.csv'):
    """
    Predict FEN notations for all test images and save the details to a CSV file.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (data.DataLoader): DataLoader for the test dataset.
        csv_path (str, optional): Path to save the CSV file. Defaults to 'all_test_predictions.csv'.
    """
    model.eval()
    image_paths = dataloader.dataset.image_paths
    results = []

    with torch.no_grad():
        for image_idx, image_path in enumerate(tqdm(image_paths, desc='Predicting Test Set')):
            try:
                fen = predict_fen(model, image_path, transform=val_transform)
                ground_truth_fen = os.path.splitext(os.path.basename(image_path))[0]
                results.append({
                    'Image_Name': os.path.basename(image_path),
                    'Ground_Truth_FEN': ground_truth_fen,
                    'Predicted_FEN': fen
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'Image_Name': os.path.basename(image_path),
                    'Ground_Truth_FEN': os.path.splitext(os.path.basename(image_path))[0],
                    'Predicted_FEN': 'Error'
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved all test predictions to {csv_path}")

# ===============================
# Visualization and Explanation Functions
# ===============================

def visualize_predictions_and_explain(model: nn.Module, dataloader: data.DataLoader, num_images: int = 50, csv_path: str = 'predictions.csv'):
    """
    Visualize sample predictions alongside actual labels, generate and save explanations, and save predictions to a CSV file.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (data.DataLoader): DataLoader for the dataset.
        num_images (int, optional): Number of images to process for visualization and explanations. Defaults to 50.
        csv_path (str, optional): Path to save the CSV file. Defaults to 'predictions.csv'.
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(20, 20))

    # Access image paths from the dataset
    image_paths = dataloader.dataset.image_paths
    total_images = len(image_paths)

    # Initialize a list to store results
    results = []

    # Initialize a dictionary to collect predicted labels per image
    image_pred_labels = {}

    # Create directory for saving explanations
    os.makedirs(EXPLAINABILITY_SAVE_PATH, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc='Processing')):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(inputs.size()[0]):
                grid_idx = images_so_far
                image_idx = grid_idx // 64  # Correctly map grid index to image index

                if image_idx >= num_images:
                    break  # Process only the specified number of images

                image_path = image_paths[image_idx]
                image_name = os.path.basename(image_path)

                if image_idx not in image_pred_labels:
                    image_pred_labels[image_idx] = []

                image_pred_labels[image_idx].append(preds[i])

                # Visualization: show the first grid of each image
                if grid_idx % 64 == 0 and image_idx < num_images:
                    ax = plt.subplot((num_images // 4) + 1, 4, (image_idx % num_images) + 1)
                    ax.axis('off')
                    pred_label = preds[i]
                    true_label = labels[i]
                    title_color = 'green' if pred_label == true_label else 'red'
                    ax.set_title(
                        f'Image: {image_name}\nPred: {LABEL_TO_FEN_SHORT.get(pred_label, "Unknown")}\nAct: {LABEL_TO_FEN_SHORT.get(true_label, "Unknown")}',
                        color=title_color)
                    img = inputs.cpu().data[i].numpy().transpose((1, 2, 0))
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                  np.array([0.485, 0.456, 0.406]), 0, 1)
                    plt.imshow(img)

                images_so_far += 1

                if image_idx >= num_images:
                    break  # Exit after processing the desired number of images

    plt.tight_layout()
    visualization_path = os.path.join(MODEL_SAVE_PATH, 'predictions_visualization.png')
    plt.savefig(visualization_path)
    plt.close()
    print(f"Saved predictions visualization to {visualization_path}")

    # Now, reconstruct FENs for the processed images
    for image_idx in range(num_images):
        image_path = image_paths[image_idx]
        image_name = os.path.basename(image_path)
        pred_labels = image_pred_labels.get(image_idx, [])
        if len(pred_labels) != 64:
            print(f"Warning: Image {image_name} has {len(pred_labels)} predictions instead of 64.")
            continue
        pred_fen = label_to_fen(pred_labels)
        # Ground truth FEN is inferred from image name (assuming image filename is the FEN)
        ground_truth_fen = os.path.splitext(image_name)[0]
        results.append({'Image_Name': image_name, 'Ground_Truth_FEN': ground_truth_fen, 'Predicted_FEN': pred_fen})

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Generate and save Integrated Gradients explanations for each image
    for image_idx in range(num_images):
        image_path = image_paths[image_idx]
        image_name = os.path.basename(image_path)
        pred_fen = df.loc[df['Image_Name'] == image_name, 'Predicted_FEN'].values[0]
        print(f"Generating explanations for {image_name} with predicted FEN: {pred_fen}")

        # Generate explanation for each grid in the image
        generate_integrated_gradients(model, image_path, val_transform, image_idx, num_images)

def generate_integrated_gradients(model: nn.Module, image_path: str, transform, image_idx: int, num_images: int):
    """
    Generate and save Integrated Gradients explanations for each grid in the image.

    Args:
        model (nn.Module): Trained PyTorch model.
        image_path (str): Path to the input image.
        transform (callable): Transformation to apply to the image.
        image_idx (int): Index of the image being processed.
        num_images (int): Total number of images to process.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Resize original image to IMAGE_SIZE to match attributions
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original_image = np.array(image_resized) / 255.0

    transformed = transform(image=np.array(image))
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    # Forward pass to get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    # Initialize IntegratedGradients
    ig = IntegratedGradients(model)

    # Define a forward function that outputs the probability for the predicted class
    def forward_func(x):
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        return probs[:, pred_label]

    # Compute attributions
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=torch.zeros_like(input_tensor),
        target=pred_label,
        return_convergence_delta=True,
        n_steps=50
    )

    # Process attributions for visualization
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    # Normalize attributions for visualization
    attributions = np.transpose(attributions, (1, 2, 0))
    attributions = np.sum(np.abs(attributions), axis=2)
    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)

    # Overlay attributions on the resized original image
    overlay = original_image.copy()
    overlay[:, :, 0] = overlay[:, :, 0] * 0.5 + attributions * 0.5
    overlay[:, :, 1] = overlay[:, :, 1] * 0.5 + attributions * 0.5
    overlay[:, :, 2] = overlay[:, :, 2] * 0.5  # Optional: Reduce the blue channel

    # Plot and save the explanation
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(f'Integrated Gradients for {image_path}')
    plt.axis('off')
    explanation_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_ig.png'
    explanation_path = os.path.join(EXPLAINABILITY_SAVE_PATH, explanation_filename)
    plt.savefig(explanation_path)
    plt.close()
    print(f"Saved Integrated Gradients explanation to {explanation_path}")

# ===============================
# Prediction Saving Function
# ===============================

def save_all_test_predictions(model: nn.Module, dataloader: data.DataLoader, csv_path: str = 'all_test_predictions.csv'):
    """
    Predict FEN notations for all test images and save the details to a CSV file.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (data.DataLoader): DataLoader for the test dataset.
        csv_path (str, optional): Path to save the CSV file. Defaults to 'all_test_predictions.csv'.
    """
    model.eval()
    image_paths = dataloader.dataset.image_paths
    results = []

    with torch.no_grad():
        for image_idx, image_path in enumerate(tqdm(image_paths, desc='Predicting Test Set')):
            try:
                fen = predict_fen(model, image_path, transform=val_transform)
                ground_truth_fen = os.path.splitext(os.path.basename(image_path))[0]
                results.append({
                    'Image_Name': os.path.basename(image_path),
                    'Ground_Truth_FEN': ground_truth_fen,
                    'Predicted_FEN': fen
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'Image_Name': os.path.basename(image_path),
                    'Ground_Truth_FEN': os.path.splitext(os.path.basename(image_path))[0],
                    'Predicted_FEN': 'Error'
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved all test predictions to {csv_path}")


# ===============================
# Main Function
# ===============================

def main():
    # Prepare DataLoaders
    try:
        # Utilize all training and test images by setting train_size and test_size to None
        train_loader, val_loader, test_loader = prepare_dataloaders(train_size=None, test_size=None, batch_size=BATCH_SIZE)
    except ValueError as e:
        print(f"DataLoader Preparation Error: {e}")
        return
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize the model
    model = get_model(pretrained=True, num_classes=LABEL_SIZE)
    print("Model initialized successfully.\n")

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.heads.head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Create directories for saving models, logs, and explanations
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EXPLAINABILITY_SAVE_PATH, exist_ok=True)

    # Train the model
    start_time = time.time()
    try:
        trained_model = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=NUM_EPOCHS,
                                    log_dir=LOG_DIR)
    except Exception as e:
        print(f"Training Error: {e}")
        return
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time) / 60:.2f} minutes\n")

    # Evaluate the model at grid-level
    try:
        evaluate_model(trained_model, test_loader)
    except Exception as e:
        print(f"Grid-Level Evaluation Error: {e}")
        return

    # Evaluate the model at image-level
    try:
        evaluate_model_image_level(trained_model, test_loader)
    except Exception as e:
        print(f"Image-Level Evaluation Error: {e}")
        return

    # Save Predictions for All Test Images
    try:
        # Save all test predictions to a CSV
        save_all_test_predictions(trained_model, test_loader, csv_path='all_test_predictions.csv')
    except Exception as e:
        print(f"Saving All Test Predictions Error: {e}")
        return

    # Visualize some predictions, generate explanations, and save to CSV
    try:
        # Adjust the number of explanations as needed
        num_images_to_explain = 50  # Adjust based on computational resources
        visualize_predictions_and_explain(trained_model, test_loader, num_images=num_images_to_explain, csv_path='predictions.csv')
    except Exception as e:
        print(f"Visualization and Explanation Error: {e}")
        return

    # Example Prediction
    try:
        # Instead of using tkinter, use command-line input to avoid 'No module named tkinter' error
        image_path = input("Enter the path to a chessboard image for FEN prediction (or press Enter to skip): ").strip()
        if image_path:
            if not os.path.isfile(image_path):
                print(f"File not found: {image_path}\n")
            else:
                fen = predict_fen(trained_model, image_path, transform=val_transform)
                print(f'Predicted FEN: {fen}\n')

                # Display the image with predicted FEN
                image = Image.open(image_path)
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.title(f'Predicted FEN: {fen}')
                plt.axis('off')
                predicted_image_path = os.path.join(MODEL_SAVE_PATH, 'predicted_image.png')
                plt.savefig(predicted_image_path)
                plt.close()
                print(f"Saved predicted image to {predicted_image_path}\n")
        else:
            print("No image path provided for prediction.\n")
    except Exception as e:
        print(f"Prediction Error: {e}")
        return

if __name__ == '__main__':
    main()
