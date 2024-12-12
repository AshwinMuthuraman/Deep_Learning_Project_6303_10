# Efficient net training
# ===============================
# Model Definition with EfficientNet-B7
# ===============================

def get_model(pretrained: bool = True, num_classes: int = cfg.OUTPUTS_A) -> nn.Module:
    """
    Initialize and return the pre-trained EfficientNet-B7 model.
    """
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )

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

# Yolo with streamlit

# ---------------------------- YOLOv7 Functions ----------------------------

@st.cache_resource
def load_yolo_model(weights_path, device):
    """
    Loads the YOLOv7 model from the provided weights file.
    """
    try:
        model = torch.load(weights_path, map_location=device)['model'].float().eval()
        st.success("YOLOv7 model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv7 model: {e}")
        return None

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Filters YOLO predictions by confidence and performs Non-Maximum Suppression (NMS).
    """
    output = []
    for image_pred in prediction:  # Per image
        # Filter by confidence
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        if not image_pred.shape[0]:
            continue

        # Multiply confidence by class probability
        image_pred[:, 5:] *= image_pred[:, 4:5]

        # Get boxes with score and class
        boxes = xywh2xyxy(image_pred[:, :4])
        scores, class_ids = image_pred[:, 5:].max(1)
        detections = torch.cat((boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)), dim=1)

        # Perform NMS
        keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thres)
        output.append(detections[keep])
    return output

def xywh2xyxy(x):
    """
    Convert YOLOv7 format from center x, center y, width, height to x1, y1, x2, y2
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def rescale_coords(img_shape, coords, original_shape):
    """
    Rescale coordinates from model to image space
    """
    gain = min(img_shape[0] / original_shape[0], img_shape[1] / original_shape[1])  # Gain
    pad = (img_shape[1] - original_shape[1] * gain) / 2, (img_shape[0] - original_shape[0] * gain) / 2  # Padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, original_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, original_shape[0])  # y1, y2
    return coords

def run_yolo_inference(model, image, conf_thres=0.25, iou_thres=0.45, device='cuda'):
    """
    Perform inference on an image using YOLOv7.
    """
    original_shape = image.shape[:2]  # Height, Width

    # Preprocess Image
    img = letterbox(image, new_shape=(640, 640))[0]  # Resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Run Inference
    with torch.no_grad():
        try:
            pred = model(img)[0]  # Run model
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
        except Exception as e:
            st.error(f"Error during YOLO inference: {e}")
            return []

    # Process Predictions
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = rescale_coords(img.shape[2:], det[:, :4], original_shape).round()
            detections.extend(det.cpu().numpy())  # Convert to numpy

    return detections

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and pads an image to fit the desired shape.
    """
    shape = img.shape[:2]  # Current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Ratio  = new / old
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # W, H
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # Padding
    dw, dh = dw // 2, dh // 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def draw_yolo_boxes(img, detections, class_names):
    """
    Draws bounding boxes and labels on the image.
    """
    for *xyxy, conf, cls in detections:
        label = f"{class_names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img

# ---------------------------- Chessboard Processing Functions ----------------------------

def process_chessboard(img, bbox, fixed_grid_size=(640, 640)):
    """
    Processes the detected chessboard bounding box to find corners and apply perspective transformation.
    Returns the image with drawn corners and the warped (aligned) chessboard image.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except ValueError as ve:
        st.warning(f"Invalid bounding box coordinates: {ve}")
        return img, None

    if y2 <= y1 or x2 <= x1:
        st.warning("Invalid bounding box dimensions.")
        return img, None

    chessboard_roi = img[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(chessboard_roi, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.warning("No contours found in the chessboard region.")
        return img, None

    # Select the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Find corners of the largest contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) != 4:
        st.warning("Detected contour does not have 4 corners.")
        return img, None

    # Order the corners in a consistent order: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [fixed_grid_size[0] - 1, 0],
        [fixed_grid_size[0] - 1, fixed_grid_size[1] - 1],
        [0, fixed_grid_size[1] - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(chessboard_roi, M, fixed_grid_size)
    except cv2.error as e:
        st.warning(f"Perspective transformation failed: {e}")
        return img, None

    # Draw the detected corners on the original image for visualization
    for point in rect:
        cv2.circle(img, (x1 + int(point[0]), y1 + int(point[1])), 5, (0, 255, 0), -1)

    return img, warped

def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find top-left and bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

# ---------------------------- Webcam Video Processor ----------------------------

class VideoProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, fen_model):
        self.yolo_model = yolo_model
        self.fen_model = fen_model
        self.fen_results = []  # Initialize as an empty list

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        original_img = img.copy()

        # YOLOv7 Inference
        detections = run_yolo_inference(self.yolo_model, img, conf_thres=YOLO_CONF_THRESHOLD, 
                                        iou_thres=YOLO_IOU_THRESHOLD, device=DEVICE)

        # Draw Bounding Boxes
        annotated_image = draw_yolo_boxes(img.copy(), detections, YOLO_CLASS_NAMES)

        # Process Each Detection
        for det_idx, det in enumerate(detections, 1):
            cls = int(det[5])
            if YOLO_CLASS_NAMES[cls] != "chessboard":
                continue  # Skip if not a chessboard

            # Apply the corner detection and perspective transform pipeline
            processed_img, warped = process_chessboard(img, det)
            if warped is not None:
                # Predict FEN
                fen = predict_fen_from_image(self.fen_model, warped)
                self.fen_results.append(fen)

                # Overlay FEN annotation near the bounding box on the annotated image
                x1, y1, x2, y2 = map(int, det[:4])
                # Position the text slightly above the top-left corner of the bounding box
                text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                cv2.putText(annotated_image, fen, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Limit the number of stored FEN results to the last 10
                if len(self.fen_results) > 10:
                    self.fen_results.pop(0)

        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
