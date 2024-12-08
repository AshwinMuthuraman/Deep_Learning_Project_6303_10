
"""
Testing Script for EfficientNet-B0 Model.
"""
import os
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from Inception import Config, ChessDataset, get_model  # Assuming the training script is named `training_script.py`

# Load Configurations
cfg = Config()

# Update model path
MODEL_PATH = "/home/ubuntu/FinalProject/Output_Data/EfficientNetB0_Model_07-12-2024_21.21.11/best_model.pth"

# Define transform for test data
test_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = ChessDataset(folder=cfg.TEST_DIR, transform=test_transform)

# Create DataLoader for test dataset
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.PIN_MEMORY
)

# Initialize the EfficientNet-B0 model
model = get_model(pretrained=False, num_classes=cfg.OUTPUTS_A)  # Use EfficientNet-B0 architecture
model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
model.to(cfg.DEVICE)
model.eval()

# Testing function
def test_model(model, dataloader):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct.double() / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return accuracy.item(), precision, recall, f1

# Evaluate the model on the test dataset
if __name__ == '__main__':
    print(f"Evaluating model on test dataset from: {cfg.TEST_DIR}")
    test_model(model, test_loader)

