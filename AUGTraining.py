import os
import random
import cv2
import numpy as np
from torchvision.datasets import Places365
from torchvision import transforms

# Paths for training dataset
train_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Train/Images"
train_augmented_path = "/home/ubuntu/FinalProject/AugmentedDataset/Train/Augmented"
train_masks_path = "/home/ubuntu/FinalProject/AugmentedDataset/Train/Masks"
os.makedirs(train_augmented_path, exist_ok=True)

# Load Places365 dataset
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])

places365_dataset = Places365(root="/home/ubuntu/AUGtest/data", split="train-standard", transform=transform, download=False)



def get_random_places_image():
    """Fetch a random Places365 image."""
    idx = random.randint(0, len(places365_dataset) - 1)
    image, label = places365_dataset[idx]
    image = image.permute(1, 2, 0).numpy() * 255  # Convert to HWC and scale to [0, 255]
    return image.astype(np.uint8)


def add_random_padding(image, final_size=(400, 400)):
    """Add either a random solid color or a Places365 background as padding."""
    if random.random() < 0.5:  # 50% chance for random solid color
        solid_color = [random.randint(0, 255) for _ in range(3)]
        background = np.full((*final_size, 3), solid_color, dtype=np.uint8)
    else:
        background = get_random_places_image()
        background = cv2.resize(background, final_size)  # Resize background to match canvas
    return background


def rotate_chessboard_without_loss(image, background, angle_range=(-30, 30), canvas_size=(400, 400)):
    """Rotate the chessboard without losing parts of it."""
    # Resize chessboard image to a random size
    chessboard_size = random.randint(200, 256)
    chessboard_image = cv2.resize(image, (chessboard_size, chessboard_size))

    # Place the chessboard at the center of the background
    canvas = background.copy()
    center_x = (canvas_size[1] - chessboard_size) // 2
    center_y = (canvas_size[0] - chessboard_size) // 2
    canvas[center_y:center_y + chessboard_size, center_x:center_x + chessboard_size] = chessboard_image

    # Rotate the entire canvas
    center = (canvas_size[1] // 2, canvas_size[0] // 2)
    angle = random.uniform(*angle_range)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_canvas = cv2.warpAffine(
        canvas,
        rotation_matrix,
        canvas_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Replace black areas with the background
    mask = cv2.inRange(rotated_canvas, (0, 0, 0), (0, 0, 0))
    rotated_canvas[mask > 0] = background[mask > 0]

    return rotated_canvas, chessboard_size


def add_random_text_around_chessboard(image, chessboard_size, canvas_size=(400, 400), num_texts_range=(6, 8)):
    """Add random numbers or letters around the chessboard."""
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define the chessboard center and bounding box
    center_x = (canvas_size[1] - chessboard_size) // 2
    center_y = (canvas_size[0] - chessboard_size) // 2
    chessboard_bbox = [
        (center_x, center_y),
        (center_x + chessboard_size, center_y + chessboard_size)
    ]

    def add_text_linearly(start, end, fixed, orientation="horizontal"):
        """Add random text along a specific axis."""
        num_texts = random.randint(*num_texts_range)
        positions = np.linspace(start, end, num_texts, dtype=int)

        for pos in positions:
            random_text = str(random.randint(0, 9)) if random.random() < 0.5 else random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            font_scale = random.uniform(0.6, 1.0)
            font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 2)

            if orientation == "horizontal":
                text_position = (pos, fixed)
            else:
                text_position = (fixed, pos)

            cv2.putText(canvas, random_text, text_position, font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

    # Top side
    add_text_linearly(center_x, center_x + chessboard_size, center_y - 20, "horizontal")
    # Bottom side
    add_text_linearly(center_x, center_x + chessboard_size, center_y + chessboard_size + 40, "horizontal")
    # Left side
    add_text_linearly(center_y, center_y + chessboard_size, center_x - 40, "vertical")
    # Right side
    add_text_linearly(center_y, center_y + chessboard_size, center_x + chessboard_size + 40, "vertical")

    return canvas


# Augment Training Images
def augment_training_images():
    train_files = [f for f in os.listdir(train_images_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    for idx, file_name in enumerate(train_files):
        file_path = os.path.join(train_images_path, file_name)
        image = cv2.imread(file_path)

        for i in range(6):  # 6 augmentations per image
            # Add padding
            background = add_random_padding(image)
            # Rotate chessboard
            augmented_image, chessboard_size = rotate_chessboard_without_loss(image, background)

            # 50% chance to add random text/numbers
            if random.random() < 0.5:
                augmented_image = add_random_text_around_chessboard(augmented_image, chessboard_size)

            # Save augmented image
            augmented_file_name = f"{os.path.splitext(file_name)[0]}_aug_{i + 1}.jpg"
            cv2.imwrite(os.path.join(train_augmented_path, augmented_file_name), augmented_image)

        if idx % 100 == 0:
            print(f"Processed {idx} / {len(train_files)} images")

augment_training_images()
print("Training dataset augmentation complete!")
