import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# U-Net Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.upconv_block(1024 + 512, 512)
        self.dec3 = self.upconv_block(512 + 256, 256)
        self.dec2 = self.upconv_block(256 + 128, 128)
        self.dec1 = self.upconv_block(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        # Decoder
        dec4 = self.dec4(torch.cat([self.upsample(bottleneck, enc4), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4, enc3), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3, enc2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2, enc1), enc1], dim=1))

        # Output
        return torch.sigmoid(self.final(dec1))

    def upsample(self, x, target):
        _, _, h, w = target.size()
        return nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


# Dataset
class ChessboardDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.file_names = os.listdir(images_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_file = self.file_names[idx]
        image = cv2.imread(os.path.join(self.images_path, image_file))

        # Remove the augmentation suffix to find the corresponding mask
        mask_file = "_".join(image_file.split("_")[:-2]) + ".jpeg"
        mask = cv2.imread(os.path.join(self.masks_path, mask_file), cv2.IMREAD_GRAYSCALE)

        # Handle missing masks
        if mask is None:
            print(f"Warning: Mask not found for {image_file}. Using a blank mask.")
            mask = np.zeros((400, 400), dtype=np.uint8)

        # Resize and normalize
        image = cv2.resize(image, (400, 400))
        mask = cv2.resize(mask, (400, 400))

        image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return image, mask


# DataLoader
def load_data(images_path, masks_path, batch_size=16, shuffle=True, num_workers=4):
    dataset = ChessboardDataset(images_path, masks_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Metrics
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.item()


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # Resize outputs to match mask size
            outputs = nn.functional.interpolate(outputs, size=(400, 400), mode='bilinear', align_corners=True)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # Resize outputs to match mask size
                outputs = nn.functional.interpolate(outputs, size=(400, 400), mode='bilinear', align_corners=True)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "unet_best_model.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss = {val_loss:.4f}")


# Paths
# Paths for augmented training dataset
train_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Train/Augmented"
train_masks_path = "/home/ubuntu/FinalProject/AugmentedDataset/Train/Masks"

# Paths for augmented validation dataset
val_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Validation/Augmented"
val_masks_path = "/home/ubuntu/FinalProject/AugmentedDataset/Validation/Masks"



# Hyperparameters
batch_size = 8  # Reduced for memory
learning_rate = 1e-4
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
model = UNet().to(device)
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoaders
train_loader = load_data(train_images_path, train_masks_path, batch_size)
val_loader = load_data(val_images_path, val_masks_path, batch_size)

# Start Training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
print("Training complete!")
