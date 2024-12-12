# -*- coding: utf-8 -*-
"""
Code to use EfficientNet-B0 and adjusted hyperparameters.
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


class Config:
    ORIGINAL_PATH = '/home/ubuntu/FinalProject/'
    DATA_DIR = os.path.join(ORIGINAL_PATH, 'dataset')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    # Adjusted hyperparameters for EfficientNet-B0
    N_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    IMAGE_SIZE = 299
    CHANNELS = 3
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SAVE_MODEL = True
    EARLY_STOPPING_PATIENCE = 5

    NUM_WORKERS = 8
    PIN_MEMORY = True

    OUTPUTS_A = 13

    FEN_TO_LABEL_DICT = {
        'p': 1, 'P': 2,
        'b': 3, 'B': 4,
        'r': 5, 'R': 6,
        'n': 7, 'N': 8,
        'q': 9, 'Q': 10,
        'k': 11, 'K': 12,
        '0': 0
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

    TIMESTAMP = datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
    NICKNAME = "EfficientNetB0_Model"
    MODEL_SAVE_PATH = os.path.join('Output_Data', f"{NICKNAME}_{TIMESTAMP}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    MODEL_SAVE_NAME = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
    PLOTS_SAVE_PATH = MODEL_SAVE_PATH
    PREDICTION_EXCEL = 'test_predictions.xlsx'


cfg = Config()


def fen_to_label(fen: str) -> List[int]:
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
    grids = []
    h, w = image.shape
    grid_h, grid_w = grid_size
    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            grid = image[i:i + grid_h, j:j + grid_w]
            if grid.shape[0] == grid_h and grid.shape[1] == grid_w:
                grids.append(grid)
    return grids


class ChessDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.image_paths = self.get_image_paths(folder)
        self.transform = transform
        self.labels = self.prepare_labels()

    def get_image_paths(self, folder: str) -> List[str]:
        extensions = ['.jpeg', '.jpg', '.JPEG', '.JPG', '.png', '.bmp', '.gif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        return image_paths

    def prepare_labels(self) -> List[List[int]]:
        labels = []
        for path in self.image_paths:
            filename = os.path.splitext(os.path.basename(path))[0]
            labels.append(fen_to_label(filename))
        return labels

    def __len__(self):
        return len(self.image_paths) * 64

    def __getitem__(self, idx):
        image_idx = idx // 64
        grid_idx = idx % 64
        img_path = self.image_paths[image_idx]
        label = self.labels[image_idx][grid_idx]

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image {img_path} could not be read. Using blank grid.")
            image_rgb = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS), dtype=np.uint8)
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grids = split_image_into_grids(image_gray, grid_size=(50, 50))
            if grid_idx >= len(grids):
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

        label = torch.tensor(label, dtype=torch.long)
        return image, label


train_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_model(pretrained: bool = True, num_classes: int = cfg.OUTPUTS_A) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier
    # EfficientNet-B0 has model.classifier as the classification head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )

    # Unfreeze classifier parameters
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(cfg.DEVICE)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pth'):
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
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


def plot_metrics(history: dict, save_path: str):
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
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = images.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train.double() / len(train_loader.dataset)

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

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy.item())
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy.item())

        epoch_time = (time.time() - start_time) / 60
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f} "
            f"| Time: {epoch_time:.2f} min"
        )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if avg_val_loss < best_val_loss and cfg.SAVE_MODEL:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with val_loss: {best_val_loss:.4f}")

    print('Training complete')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    os.makedirs(cfg.PLOTS_SAVE_PATH, exist_ok=True)
    plot_metrics(history, cfg.PLOTS_SAVE_PATH)
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return history


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float, float, float]:
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

    print(f'Accuracy on the {total} test grids: {accuracy * 100:.2f} %')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    return accuracy.item(), precision, recall, f1


def main():
    os.chdir(cfg.ORIGINAL_PATH)

    # Create a full dataset from the training directory
    full_dataset = ChessDataset(folder=cfg.TRAIN_DIR, transform=train_transform)

    # Split into train and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    test_dataset = ChessDataset(folder=cfg.TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    print(f"Number of training grids: {len(train_loader.dataset)}")
    print(f"Number of validation grids: {len(val_loader.dataset)}")

    model = get_model(pretrained=True, num_classes=cfg.OUTPUTS_A)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

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

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_NAME))
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    main()
