import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



import torch
torch.cuda.empty_cache()

import torch.nn as nn


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

# Define Dataset Class for Testing
class ChessboardTestDataset(Dataset):
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

# Load Data
def load_test_data(images_path, masks_path, batch_size=8, num_workers=4):
    dataset = ChessboardTestDataset(images_path, masks_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Metrics
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.item()

# Test Function
def test_model(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    dice_score = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Resize outputs to match mask size
            outputs = nn.functional.interpolate(outputs, size=(400, 400), mode='bilinear', align_corners=True)

            # Calculate Dice coefficient
            dice_score += dice_coefficient(outputs, masks)

            # Save predictions for visualization
            preds = (outputs > 0.5).float().cpu().numpy()
            for i in range(preds.shape[0]):
                pred_mask = (preds[i, 0] * 255).astype(np.uint8)
                file_name = f"prediction_{idx * test_loader.batch_size + i + 1}.png"
                cv2.imwrite(os.path.join(output_dir, file_name), pred_mask)

    avg_dice = dice_score / len(test_loader)
    print(f"Average Dice Coefficient on Test Dataset: {avg_dice:.4f}")

# Paths for Testing Dataset
test_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Test/Augmented"
test_masks_path = "/home/ubuntu/FinalProject/AugmentedDataset/Test/Masks"
output_dir = "/home/ubuntu/FinalProject/TestPredictions"

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_best_model.pth", map_location=device))
print("Model loaded successfully!")

# Load Test Data
test_loader = load_test_data(test_images_path, test_masks_path, batch_size=8)

# Run Testing
test_model(model, test_loader, device, output_dir)
print(f"Testing complete! Predictions saved in {output_dir}")
