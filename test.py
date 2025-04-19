import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from skimage import exposure
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, roc_curve, auc


# Define custom metrics to load the model
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

def iou_score(y_true, y_pred, smooth=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    return (intersection + smooth) / (union + smooth)

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


# Load the trained model
def load_trained_model(model_path='fcn_retinal_vessels_final.keras'):
    print(f"Loading model from {model_path}...")

    # Custom objects dictionary for loading model with custom metrics
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou_score': iou_score,
        'specificity': specificity,
        'sensitivity': sensitivity
    }

    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
    return model


# Preprocess a single image
def preprocess_image(image_path, img_height=512, img_width=512):
    print(f"Preprocessing image: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Resize image
    img = cv2.resize(img, (img_width, img_height))

    # Convert to RGB if it's grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Normalize to [0,1]
    img = img / 255.0

    return img


# Preprocess test images and their masks (if available)
def prepare_test_data(test_image_path, test_mask_path=None, img_height=512, img_width=512):
    # Process test image
    test_img = preprocess_image(test_image_path, img_height, img_width)

    # Process mask if available
    test_mask = None
    if test_mask_path and os.path.exists(test_mask_path):
        mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height))
        mask = (mask > 127).astype(np.float32)  # Binarize mask
        test_mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

    return test_img, test_mask


# Make prediction
def predict_segmentation(model, test_img):
    print("Generating prediction...")

    # Expand dimensions to create a batch of size 1
    test_img_batch = np.expand_dims(test_img, axis=0)

    # Get prediction
    prediction = model.predict(test_img_batch)[0]

    return prediction


# Calculate metrics if ground truth is available
def calculate_metrics(true_mask, pred_mask, threshold=0.5):
    if true_mask is None:
        print("No ground truth mask provided. Skipping metrics calculation.")
        return None

    # Binarize prediction using threshold
    pred_binary = (pred_mask > threshold).astype(np.float32)

    # Flatten masks
    true_flat = true_mask.flatten()
    pred_flat = pred_binary.flatten()

    # Calculate metrics
    dice = dice_coefficient(true_flat, pred_flat).numpy()
    iou = iou_score(true_flat, pred_flat).numpy()

    # Calculate F1 score
    f1 = f1_score(true_flat > 0.5, pred_flat > 0.5)

    # Calculate Jaccard index using sklearn
    jaccard = jaccard_score(true_flat > 0.5, pred_flat > 0.5)

    # Calculate sensitivity and specificity
    sens = sensitivity(true_flat, pred_flat).numpy()
    spec = specificity(true_flat, pred_flat).numpy()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_flat, pred_mask.flatten())
    roc_auc = auc(fpr, tpr)

    metrics = {
        'Dice Coefficient': dice,
        'IoU Score': iou,
        'F1 Score': f1,
        'Jaccard Index': jaccard,
        'Sensitivity': sens,
        'Specificity': spec,
        'ROC AUC': roc_auc
    }

    return metrics, fpr, tpr


# Visualize results
def visualize_results(original_img, true_mask, pred_mask, metrics=None, save_path=None):
    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot original image
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    # Plot prediction
    plt.subplot(132)
    plt.title('Predicted Segmentation')
    plt.imshow(original_img)
    plt.imshow(pred_mask[:, :, 0], alpha=0.5, cmap='jet')
    plt.axis('off')

    # If ground truth is available, plot it
    if true_mask is not None:
        plt.subplot(133)
        plt.title('Ground Truth Mask')
        plt.imshow(original_img)
        plt.imshow(true_mask[:, :, 0], alpha=0.5, cmap='jet')
        plt.axis('off')

    plt.tight_layout()

    # If metrics are available, add them to the figure
    if metrics:
        plt.figtext(0.5, 0.01, f"Dice: {metrics['Dice Coefficient']:.4f}, IoU: {metrics['IoU Score']:.4f}, "
                               f"F1: {metrics['F1 Score']:.4f}, Sensitivity: {metrics['Sensitivity']:.4f}, "
                               f"Specificity: {metrics['Specificity']:.4f}", ha='center')

    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    else:
        plt.show()

    plt.close()


# Create overlay between prediction and ground truth (if available)
def create_overlap_visualization(original_img, true_mask, pred_mask, threshold=0.5, save_path=None):
    # Convert prediction to binary
    pred_binary = (pred_mask > threshold).astype(np.uint8)

    # Create RGB version of prediction (red channel)
    pred_rgb = np.zeros((*pred_binary.shape[:2], 3), dtype=np.uint8)
    pred_rgb[:, :, 0] = pred_binary[:, :, 0] * 255  # Red channel

    # If ground truth is available, create visualization
    if true_mask is not None:
        # Convert true mask to binary and create green channel
        true_binary = (true_mask > threshold).astype(np.uint8)
        true_rgb = np.zeros((*true_binary.shape[:2], 3), dtype=np.uint8)
        true_rgb[:, :, 1] = true_binary[:, :, 0] * 255  # Green channel


        overlap = np.zeros((*pred_binary.shape[:2], 3), dtype=np.uint8)
        overlap[:, :, 0] = pred_binary[:, :, 0] * 255  # Red channel (predictions)
        overlap[:, :, 1] = true_binary[:, :, 0] * 255  # Green channel (ground truth)

        # Create figure
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(original_img)
        plt.axis('off')


        plt.subplot(132)
        plt.title('Prediction Segmentation Image')
        plt.imshow(original_img)
        plt.imshow(overlap, alpha=0.5)
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overlap visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()
    else:
        # Just show prediction if no ground truth
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(original_img)
        plt.axis('off')

        plt.subplot(122)
        plt.title('Prediction')
        plt.imshow(original_img)
        plt.imshow(pred_rgb, alpha=0.5)
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction only visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


# Apply post-processing to enhance prediction
def postprocess_prediction(pred_mask, threshold=0.5):
    # Threshold the prediction
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # Remove small connected components (noise)
    num_labels, labels = cv2.connectedComponents(binary_mask[:, :, 0])

    # Calculate area of each component
    for label in range(1, num_labels):
        component_size = np.sum(labels == label)
        # If component is too small, remove it
        if component_size < 50:  # Adjust this threshold as needed
            binary_mask[labels == label] = 0

    # Morphological operations to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert back to float and expand dimensions
    processed_mask = binary_mask.astype(np.float32)
    if len(processed_mask.shape) == 2:
        processed_mask = np.expand_dims(processed_mask, axis=-1)

    return processed_mask


# Main testing function
def test_fcn_model(image_path, mask_path=None, model_path='fcn_retinal_vessels_final.keras', output_dir='test_results'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename from path
    image_name = os.path.basename(image_path).split('.')[0]

    # Load model
    model = load_trained_model(model_path)

    # Prepare test data
    test_img, test_mask = prepare_test_data(image_path, mask_path)

    # Get prediction
    pred_mask = predict_segmentation(model, test_img)

    # Post-process prediction
    processed_mask = postprocess_prediction(pred_mask)

    # Calculate metrics if ground truth is available
    metrics = None
    if test_mask is not None:
        metrics, fpr, tpr = calculate_metrics(test_mask, pred_mask)

        # Print metrics
        print("\nSegmentation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["ROC AUC"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f"{image_name}_roc_curve.png"))
        plt.close()

    # Visualize results
    visualize_results(test_img, test_mask, pred_mask, metrics,
                      save_path=os.path.join(output_dir, f"{image_name}_segmentation.png"))

    # Create overlap visualization
    create_overlap_visualization(test_img, test_mask, pred_mask,
                                 save_path=os.path.join(output_dir, f"{image_name}_overlap.png"))

    # Save prediction as image
    pred_image = (pred_mask[:, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_prediction.png"), pred_image)

    # Return the results
    return test_img, test_mask, pred_mask, metrics


# Function to test multiple images
def test_multiple_images(test_dir, mask_dir=None, model_path='fcn_retinal_vessels_final.keras',
                         output_dir='test_results'):
    # Get list of test images
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize metrics storage
    all_metrics = []

    # Process each image
    for img_file in test_images:
        image_path = os.path.join(test_dir, img_file)

        # Look for corresponding mask if mask_dir provided
        mask_path = None
        if mask_dir:
            # Try different potential mask naming patterns
            potential_mask_names = [
                img_file,  # Same name
                img_file.replace('.jpg', '.png').replace('.jpeg', '.png'),  # Different extension
                img_file.replace('image', 'mask'),  # Replace "image" with "mask"
                img_file.split('.')[0] + '_mask.png'  # Add "_mask" suffix
            ]

            for mask_name in potential_mask_names:
                potential_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break

        print(f"\nProcessing image: {img_file}")
        if mask_path:
            print(f"Found corresponding mask: {os.path.basename(mask_path)}")
        else:
            print("No corresponding mask found")

        # Test the image
        _, _, _, metrics = test_fcn_model(image_path, mask_path, model_path, output_dir)

        # Store metrics if available
        if metrics:
            metrics['Image'] = img_file
            all_metrics.append(metrics)

    # If we have metrics, create a summary
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df.set_index('Image')

        # Calculate average metrics
        avg_metrics = metrics_df.mean()
        metrics_df.loc['Average'] = avg_metrics

        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(output_dir, 'all_test_metrics.csv'))

        # Print average metrics
        print("\nAverage Metrics Across All Test Images:")
        for metric_name, metric_value in avg_metrics.items():
            print(f"Average {metric_name}: {metric_value:.4f}")

    print(f"\nAll test results saved to {output_dir}")


# Example usage function
def main():
    # Path to your trained model
    model_path = 'fcn_retinal_vessels_final.keras'

    # Set paths to your test data
    # Example with a single image
    single_test_image = 'Data/test/image/0.png'  # Replace with your test image path
    single_test_mask = 'Data/test/mask/0.png'  # Replace with corresponding mask (if available)

    # Test with a single image
    if os.path.exists(single_test_image):
        print(f"Testing with single image: {single_test_image}")
        test_fcn_model(single_test_image,
                       single_test_mask if os.path.exists(single_test_mask) else None,
                       model_path,
                       'single_test_results')
    else:
        print(f"Test image not found: {single_test_image}")

    # Test with multiple images
    test_image_dir = 'Data/test/image'  # Replace with your test directory
    test_mask_dir = 'Data/test/mask'  # Replace with your mask directory

    if os.path.exists(test_image_dir):
        print(f"\nTesting with multiple images from: {test_image_dir}")
        test_multiple_images(test_image_dir,
                             test_mask_dir if os.path.exists(test_mask_dir) else None,
                             model_path,
                             'batch_test_results')
    else:
        print(f"Test directory not found: {test_image_dir}")


if __name__ == "__main__":
    main()