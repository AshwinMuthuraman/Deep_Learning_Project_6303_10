import os
import random
import cv2
import numpy as np

# Paths for validation dataset
validation_images_path = "/home/ubuntu/FinalProject/AugmentedDataset/Validation/Images"
validation_augmented_path = "/home/ubuntu/FinalProject/AugmentedDataset/Validation/Augmented"
os.makedirs(validation_augmented_path, exist_ok=True)

def add_solid_color_padding(image, final_size=(400, 400)):
    """Add random solid color padding."""
    solid_color = [random.randint(0, 255) for _ in range(3)]
    background = np.full((*final_size, 3), solid_color, dtype=np.uint8)
    return background

def rotate_chessboard_without_loss(image, background, angle_range=(-15, 15), canvas_size=(400, 400)):
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

def augment_validation_images():
    # Fetch all validation image files
    validation_files = [f for f in os.listdir(validation_images_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    print(f"Number of validation files to process: {len(validation_files)}")

    for idx, file_name in enumerate(validation_files):
        file_path = os.path.join(validation_images_path, file_name)
        image = cv2.imread(file_path)

        for i in range(3):  # 3 augmentations per image
            # Add solid color padding
            background = add_solid_color_padding(image)
            # Rotate chessboard (±15°)
            augmented_image, chessboard_size = rotate_chessboard_without_loss(image, background, angle_range=(-15, 15))

            # 50% chance to add random text/numbers
            if random.random() < 0.5:
                augmented_image = add_random_text_around_chessboard(augmented_image, chessboard_size)

            # Save augmented image
            augmented_file_name = f"{os.path.splitext(file_name)[0]}_aug_{i + 1}.jpg"
            cv2.imwrite(os.path.join(validation_augmented_path, augmented_file_name), augmented_image)

        if idx % 100 == 0:
            print(f"Processed {idx} / {len(validation_files)} validation images")

    print("Validation dataset augmentation complete!")

# Run the augmentation process
if __name__ == "__main__":
    augment_validation_images()
