
import os
import glob
import random
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class Config:
    ORIGINAL_PATH = '/home/ubuntu/FinalProject/'
    DATA_DIR = os.path.join(ORIGINAL_PATH, 'dataset')  # Assuming dataset folder is in current directory
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    IMAGE_SIZE = 224
    CHANNELS = 3

    NUM_WORKERS = 8  
    PIN_MEMORY = True  


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


cfg = Config()



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



class ChessDataset(Dataset):
    """
    Custom Dataset for Chess Piece Classification.
    Processes images on-the-fly with data augmentation.
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

        image = cv2.imread(img_path)
        if image is None:
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
            transformed = self.transform(image=np.array(pil_image))
            image = transformed['image']
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image = transform(pil_image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


train_transform = A.Compose([
    A.Resize(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE),  # Ensure resizing
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(p=0.3),
    A.CLAHE(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE),  # Ensure resizing
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])



def prepare_dataloaders(folder: str, transform, batch_size: int = cfg.BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    dataset = ChessDataset(folder=folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
                            pin_memory=cfg.PIN_MEMORY)
    return dataloader



def main():
    os.chdir(cfg.ORIGINAL_PATH)

   
    print("Starting data preparation...")

    print("Creating training DataLoader with data augmentation...")
    train_loader = prepare_dataloaders(folder=cfg.TRAIN_DIR, transform=train_transform, batch_size=cfg.BATCH_SIZE, shuffle=True)

    print("Creating validation DataLoader without data augmentation...")
    val_loader = prepare_dataloaders(folder=cfg.TRAIN_DIR, transform=val_transform, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("DataLoaders are ready.")

    for images, labels in train_loader:
        print(f"Batch of training images shape: {images.shape}")
        print(f"Batch of training labels shape: {labels.shape}")
        break  

    for images, labels in val_loader:
        print(f"Batch of validation images shape: {images.shape}")
        print(f"Batch of validation labels shape: {labels.shape}")
        break  
    print("Data augmentation and loading demonstration complete.")

if __name__ == '__main__':
    main()
