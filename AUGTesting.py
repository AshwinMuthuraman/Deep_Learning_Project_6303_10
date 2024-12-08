import os
import random
import cv2
import numpy as np
from torchvision.datasets import Places365
from torchvision import transforms

# Paths for test dataset
test_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Test/Images"
test_augmented_path = "/home/ubuntu/FinalProject/AugmentedDataset/Test/Augmented"
os.makedirs(test_augmented_path, exist_ok=True)

# Load Places365 dataset with absolute path
transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Resize images to desired size
    transforms.ToTensor()
])

places365_dataset = Places365(root="/home/ubuntu/AUGtest/data", split="train-standard", transform=transform, download=False)

# Filter indoor categories using the absolute path for the categories file
categories_file = "/home/ubuntu/AUGtest/data/categories_places365.txt"
with open(categories_file, "r") as f:
    categories = [line.strip().split(" ")[0][3:] for line in f.readlines()]

indoor_categories = [
    "airport_terminal", "art_gallery", "bakery", "bar", "bedroom", "bookstore", "classroom",
    "cloister", "conference_room", "dining_room", "gameroom", "kitchen", "library",
    "living_room", "lobby", "museum", "nursery", "office", "restaurant", "shop",
    "staircase", "supermarket", "train_interior", "waiting_room", "wine_cellar"
]
indoor_indices = [categories.index(cat) for cat in indoor_categories if cat in categories]

def get_random_indoor_image():
    """Fetch a random indoor Places365 image."""
    while True:
        idx = random.randint(0, len(places365_dataset) - 1)
        _, label = places365_dataset[idx]
        if label in indoor_indices:
            image, _ = places365_dataset[idx]
            image = image.permute(1, 2, 0).numpy() * 255  # Convert to HWC and scale to [0, 255]
            return image.astype(np.uint8)

def add_solid_color_padding(image, final_size=(400, 400)):
    """Add random solid color padding."""
    solid_color = [random.randint(0, 255) for _ in range(3)]
    background = np.full((*final_size, 3), solid_color, dtype=np.uint8)
    return background

def rotate_chessboard_without_loss(image, background, angle_range=(-10, 10), canvas_size=(400, 400)):
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

def add_random_text_around_chessboard(image, chessboard_size, canvas_size=(400, 400)):
    """Add random text (letters or numbers) around the chessboard."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # White text

    # Define text positions
    padding_top = (canvas_size[0] - chessboard_size) // 2
    padding_left = (canvas_size[1] - chessboard_size) // 2

    # Add random text on all four sides
    for i in range(6):  # 6 items on each side
        text_top = str(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        text_left = str(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        text_bottom = str(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        text_right = str(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))

        # Top side
        x_top = padding_left + i * (chessboard_size // 6)
        y_top = padding_top - 10
        cv2.putText(image, text_top, (x_top, y_top), font, font_scale, color, thickness)

        # Bottom side
        x_bottom = padding_left + i * (chessboard_size // 6)
        y_bottom = padding_top + chessboard_size + 20
        cv2.putText(image, text_bottom, (x_bottom, y_bottom), font, font_scale, color, thickness)

        # Left side
        x_left = padding_left - 20
        y_left = padding_top + i * (chessboard_size // 6)
        cv2.putText(image, text_left, (x_left, y_left), font, font_scale, color, thickness)

        # Right side
        x_right = padding_left + chessboard_size + 10
        y_right = padding_top + i * (chessboard_size // 6)
        cv2.putText(image, text_right, (x_right, y_right), font, font_scale, color, thickness)

    return image

def augment_test_images():
    # Fetch all test image files
    test_files = [f for f in os.listdir(test_images_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    print(f"Number of test files to process: {len(test_files)}")

    for idx, file_name in enumerate(test_files):
        file_path = os.path.join(test_images_path, file_name)
        image = cv2.imread(file_path)

        for i in range(2):  # 2 augmentations per image
            # 50% chance for solid color or Places365 indoor image
            if random.random() < 0.5:
                background = add_solid_color_padding(image)
            else:
                background = get_random_indoor_image()

            # Rotate chessboard (±10°)
            augmented_image, chessboard_size = rotate_chessboard_without_loss(image, background, angle_range=(-10, 10))

            # 50% chance to add random text/numbers
            if random.random() < 0.5:
                augmented_image = add_random_text_around_chessboard(augmented_image, chessboard_size)

            # Save augmented image
            augmented_file_name = f"{os.path.splitext(file_name)[0]}_aug_{i + 1}.jpg"
            cv2.imwrite(os.path.join(test_augmented_path, augmented_file_name), augmented_image)

        if idx % 100 == 0:
            print(f"Processed {idx} / {len(test_files)} test images")

    print("Test dataset augmentation complete!")

# Run the augmentation process
if __name__ == "__main__":
    augment_test_images()
