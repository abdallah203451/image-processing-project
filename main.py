import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Concatenate, Activation, Input
from tensorflow import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, f1_score

# Directories
train_image_dir = "IPProjectDataset24/train_data/images"
train_label_dir = "IPProjectDataset24/train_data/labels"
val_image_dir = "IPProjectDataset24/val_data/images"
val_label_dir = "IPProjectDataset24/val_data/labels"
test_image_dir = "IPProjectDataset24/test_data/images"

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 8
BATCH_SIZE = 16
EPOCHS = 40

# Label Color Map
LABEL_COLORS = {
    (0, 0, 0): 0,      # Background clutter
    (128, 0, 0): 1,    # Building
    (128, 64, 128): 2, # Road
    (0, 128, 0): 3,    # Tree
    (128, 128, 0): 4,  # Low vegetation
    (64, 0, 128): 5,   # Moving car
    (192, 0, 192): 6,  # Static car
    (64, 64, 0): 7     # Human
}

def map_colors_to_labels(mask):
    """Map RGB colors to label indices."""
    mask = tf.numpy_function(
        func=lambda x: np.array([[LABEL_COLORS[tuple(pixel)] for pixel in row] for row in x], dtype=np.uint8),
        inp=[mask],
        Tout=tf.uint8
    )
    return mask

def process_path(image_path, label_path):
    """Load and preprocess image and label paths."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0

    mask = tf.io.read_file(label_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = map_colors_to_labels(mask)

    return img, mask

def load_dataset(image_dir, label_dir):
    """Load dataset from directories."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

def unet_model(input_shape):
    """Build U-Net model."""
    inputs = Input(input_shape)

    # Encoder
    def encoder_block(inputs, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        p = MaxPool2D((2, 2))(x)
        return x, p

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = Conv2D(1024, (3, 3), padding="same", activation="relu")(p4)

    # Decoder
    def decoder_block(inputs, skip, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip])
        x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
        return x

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(NUM_CLASSES, (1, 1), activation="softmax")(d4)
    return Model(inputs, outputs)

# Load datasets
train_dataset = load_dataset(train_image_dir, train_label_dir)
val_dataset = load_dataset(val_image_dir, val_label_dir)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    metrics = {
        "IoU": jaccard_score(y_true_flat, y_pred_flat, average="macro"),
        "Accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "Precision": precision_score(y_true_flat, y_pred_flat, average="macro"),
        "Recall": recall_score(y_true_flat, y_pred_flat, average="macro"),
        "F1 Score": f1_score(y_true_flat, y_pred_flat, average="macro")
    }
    return metrics

# Build and compile U-Net model
model = unet_model((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Save model
model.save("unet_model.h5")

# Evaluate on validation dataset
for images, masks in val_dataset.take(1):
    preds = model.predict(images)
    preds = np.argmax(preds, axis=-1)
    masks = masks.numpy()

    metrics = calculate_metrics(masks, preds)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")