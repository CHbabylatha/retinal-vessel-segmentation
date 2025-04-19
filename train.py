import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score, jaccard_score
import cv2
import pandas as pd
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
image_dir = 'Data/train/image'
mask_dir = 'Data/train/mask'

# Image dimensions
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3


# FCN model architecture
def build_fcn_model(input_shape):
    inputs = Input(input_shape)

    # Encoder path (contracting)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder path (expanding)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Data preprocessing
def preprocess_data():
    print("Loading and preprocessing data...")

    # Get filenames
    image_files = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                          fname.endswith('.png') or fname.endswith('.jpg')])
    mask_files = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if
                         fname.endswith('.png') or fname.endswith('.jpg')])

    # Check if we have matching files
    assert len(image_files) == len(mask_files), "Number of images and masks do not match"
    print(f"Found {len(image_files)} image-mask pairs")

    # Load and preprocess images and masks
    images = []
    masks = []

    for img_path, mask_path in zip(image_files, mask_files):
        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = img / 255.0  # Normalize to [0,1]
        images.append(img)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = (mask > 127).astype(np.float32)  # Binarize and convert to float32
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)

    return np.array(images), np.array(masks)


# Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)


# IoU metric (Jaccard index)
def iou_score(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    return (intersection + smooth) / (union + smooth)


# Create custom metrics for tracking
def specificity(y_true, y_pred):
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = tf.reduce_sum(y_true_neg * y_pred_neg)
    possible_negatives = tf.reduce_sum(y_true_neg)
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


# Training function
def train_fcn_model():
    # Load and preprocess data
    images, masks = preprocess_data()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build model
    model = build_fcn_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Compile model with appropriate metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            dice_coefficient,
            iou_score,
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            sensitivity,
            specificity,
            AUC(name='auc')
        ]
    )

    # Model summary
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint(
        'fcn_retinal_vessels.keras',  # Changed to .keras extension
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_dice_coefficient',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Initialize metrics tracking
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_sensitivity': [],
        'val_sensitivity': [],
        'train_specificity': [],
        'val_specificity': [],
        'train_auc': [],
        'val_auc': [],
        'f1_score': [],
        'roc_auc': []
    }

    # Training parameters
    batch_size = 8
    epochs = 50

    # Train model and track metrics
    print("Training model...")

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    if steps_per_epoch == 0:
        steps_per_epoch = 1
    if validation_steps == 0:
        validation_steps = 1

    # Manual training loop to track all metrics
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Training phase
        train_losses = []
        train_dices = []
        train_ious = []
        train_accs = []
        train_precisions = []
        train_recalls = []
        train_sensitivities = []
        train_specificities = []
        train_aucs = []

        for i in range(steps_per_epoch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train_shuffled))

            batch_x = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]

            metrics = model.train_on_batch(batch_x, batch_y, return_dict=True)

            train_losses.append(metrics['loss'])
            train_dices.append(metrics['dice_coefficient'])
            train_ious.append(metrics['iou_score'])
            train_accs.append(metrics['accuracy'])
            train_precisions.append(metrics['precision'])
            train_recalls.append(metrics['recall'])
            train_sensitivities.append(metrics['sensitivity'])
            train_specificities.append(metrics['specificity'])
            train_aucs.append(metrics['auc'])

            # Print progress
            if (i + 1) % 10 == 0 or i == steps_per_epoch - 1:
                print(
                    f"Step {i + 1}/{steps_per_epoch} - loss: {np.mean(train_losses):.4f} - dice: {np.mean(train_dices):.4f}")

        # Validation phase
        val_losses = []
        val_dices = []
        val_ious = []
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_sensitivities = []
        val_specificities = []
        val_aucs = []

        val_y_true = []
        val_y_pred = []

        for i in range(validation_steps):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_val))

            batch_x = X_val[start_idx:end_idx]
            batch_y = y_val[start_idx:end_idx]

            batch_pred = model.predict_on_batch(batch_x)
            metrics = model.test_on_batch(batch_x, batch_y, return_dict=True)

            val_losses.append(metrics['loss'])
            val_dices.append(metrics['dice_coefficient'])
            val_ious.append(metrics['iou_score'])
            val_accs.append(metrics['accuracy'])
            val_precisions.append(metrics['precision'])
            val_recalls.append(metrics['recall'])
            val_sensitivities.append(metrics['sensitivity'])
            val_specificities.append(metrics['specificity'])
            val_aucs.append(metrics['auc'])

            # Store true and predicted values for ROC and F1
            val_y_true.extend(batch_y.flatten())
            val_y_pred.extend(batch_pred.flatten())

        # Calculate F1-score and ROC AUC
        val_y_true_binary = np.array(val_y_true) > 0.5
        val_y_pred_binary = np.array(val_y_pred) > 0.5
        f1 = f1_score(val_y_true_binary, val_y_pred_binary)

        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(val_y_true, val_y_pred)
            roc_auc_value = auc(fpr, tpr)
        except:
            roc_auc_value = 0.5  # Default value if calculation fails

        # Update metrics history
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(np.mean(train_losses))
        metrics_history['val_loss'].append(np.mean(val_losses))
        metrics_history['train_dice'].append(np.mean(train_dices))
        metrics_history['val_dice'].append(np.mean(val_dices))
        metrics_history['train_iou'].append(np.mean(train_ious))
        metrics_history['val_iou'].append(np.mean(val_ious))
        metrics_history['train_accuracy'].append(np.mean(train_accs))
        metrics_history['val_accuracy'].append(np.mean(val_accs))
        metrics_history['train_precision'].append(np.mean(train_precisions))
        metrics_history['val_precision'].append(np.mean(val_precisions))
        metrics_history['train_recall'].append(np.mean(train_recalls))
        metrics_history['val_recall'].append(np.mean(val_recalls))
        metrics_history['train_sensitivity'].append(np.mean(train_sensitivities))
        metrics_history['val_sensitivity'].append(np.mean(val_sensitivities))
        metrics_history['train_specificity'].append(np.mean(train_specificities))
        metrics_history['val_specificity'].append(np.mean(val_specificities))
        metrics_history['train_auc'].append(np.mean(train_aucs))
        metrics_history['val_auc'].append(np.mean(val_aucs))
        metrics_history['f1_score'].append(f1)
        metrics_history['roc_auc'].append(roc_auc_value)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {np.mean(train_losses):.4f} - Val Loss: {np.mean(val_losses):.4f}")
        print(f"Train Dice: {np.mean(train_dices):.4f} - Val Dice: {np.mean(val_dices):.4f}")
        print(f"Train IoU: {np.mean(train_ious):.4f} - Val IoU: {np.mean(val_ious):.4f}")
        print(f"F1 Score: {f1:.4f} - ROC AUC: {roc_auc_value:.4f}")

        # Check early stopping condition
        best_val_dice = max(metrics_history['val_dice'])
        if epoch > 0 and epoch - np.argmax(metrics_history['val_dice']) >= early_stopping.patience:
            print(f"Early stopping triggered. Best val_dice: {best_val_dice:.4f}")
            break

    # Save the model
    model.save('fcn_retinal_vessels_final.keras')  # Changed to .keras extension

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_history)

    # Save metrics to CSV
    metrics_df.to_csv('fcn_training_metrics.csv', index=False)

    # Print final metrics
    print("\nTraining completed!")
    print(f"Best validation Dice coefficient: {best_val_dice:.4f}")

    return model, metrics_df


# Plot metrics
def plot_metrics(metrics_df):
    # Create output directory for plots
    os.makedirs('metric_plots', exist_ok=True)

    # List of metrics to plot
    metrics = [
        ('Dice Coefficient (DSC)', 'val_dice'),
        ('IoU Score', 'val_iou'),
        ('Accuracy', 'val_accuracy'),
        ('Precision', 'val_precision'),
        ('Recall', 'val_recall'),
        ('Sensitivity', 'val_sensitivity'),
        ('Specificity', 'val_specificity'),
        ('ROC AUC', 'val_auc'),
        ('F1 Score', 'f1_score'),
        ('ROC AUC', 'roc_auc')
    ]

    # Create individual plots for each metric
    for metric_name, metric_key in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df[metric_key], 'b-', label=f'Validation {metric_name}')

        # If there's a corresponding training metric, plot it too
        train_key = f"train_{metric_key.replace('val_', '')}" if 'val_' in metric_key else None
        if train_key in metrics_df.columns:
            plt.plot(metrics_df['epoch'], metrics_df[train_key], 'r-', label=f'Training {metric_name}')

        plt.title(f'Epoch vs {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

        # Save the plot
        filename = f"metric_plots/epoch_vs_{metric_key.replace('val_', '')}.png"
        plt.savefig(filename)
        plt.close()

    # Create a summary plot with all validation metrics
    plt.figure(figsize=(12, 8))
    for metric_name, metric_key in metrics:
        if 'val_' in metric_key or metric_key in ['f1_score', 'roc_auc']:
            plt.plot(metrics_df['epoch'], metrics_df[metric_key], label=metric_name)

    plt.title('All Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('metric_plots/all_validation_metrics.png')
    plt.close()

    print(f"Plots saved in 'metric_plots' directory")


# Function to create a tabular output
def create_metrics_table(metrics_df):
    # Create a formatted table to display all metrics
    table_df = metrics_df[['epoch', 'val_dice', 'f1_score', 'val_accuracy',
                           'val_sensitivity', 'val_specificity', 'roc_auc',
                           'val_precision', 'val_recall', 'val_iou']]

    # Rename columns for better readability
    table_df = table_df.rename(columns={
        'epoch': 'Epoch',
        'val_dice': 'Dice Coefficient (DSC)',
        'f1_score': 'F1 Score',
        'val_accuracy': 'Accuracy',
        'val_sensitivity': 'Sensitivity',
        'val_specificity': 'Specificity',
        'roc_auc': 'ROC AUC',
        'val_precision': 'Precision',
        'val_recall': 'Recall',
        'val_iou': 'IoU'
    })

    # Save the table to CSV
    table_df.to_csv('metrics_table.csv', index=False)

    return table_df


# Main function
def main():
    start_time = datetime.now()
    print(f"Starting at: {start_time}")

    # Train model
    model, metrics_df = train_fcn_model()

    # Plot metrics
    plot_metrics(metrics_df)

    # Create and print metrics table
    table_df = create_metrics_table(metrics_df)
    print("\nMetrics Table:")
    print(table_df.to_string(index=False))

    end_time = datetime.now()
    print(f"\nFinished at: {end_time}")
    print(f"Total runtime: {end_time - start_time}")


if __name__ == "__main__":
    main()