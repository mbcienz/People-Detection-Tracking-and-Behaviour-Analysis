import os
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TestDataset
from nets import PARMultiTaskNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Function to test the model
def test_model(model, dataloader, device, thresholds):
    """
    Tests the given model on the provided dataloader and computes evaluation metrics.

    Args:
        model: The trained multi-task model to be evaluated.
        dataloader: DataLoader for the test dataset.
        device: Device on which the computations will be performed (CPU or GPU).
        thresholds (dict): Dictionary of thresholds for each task.

    Returns:
        metrics (dict): A dictionary containing evaluation metrics (accuracy, precision, recall, F1-score) for each task.
    """
    model.eval()  # Set the model to evaluation mode

    # Dictionaries to store true labels and predictions for each task
    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in tqdm(dataloader, desc=f"Test..."):
            images, labels = images.to(device), labels.to(device)

            # Mask for valid samples (labels not equal to -1)
            masks = labels >= 0

            outputs = model(images)  # Forward pass through the model

            # Process predictions and labels for each task
            for task in ["gender", "bag", "hat"]:
                task_index = ["gender", "bag", "hat"].index(task)

                # Get the threshold for the current task
                threshold = thresholds[task]

                # Binary predictions (threshold = 0.5)
                preds = (torch.sigmoid(outputs[task]) > threshold).int()

                # Filter valid samples using the mask
                valid_preds = preds[masks[:, task_index]].cpu().numpy()
                valid_labels = labels[masks[:, task_index], task_index].cpu().numpy()

                # Append valid predictions and labels
                all_predictions[task].extend(valid_preds)
                all_labels[task].extend(valid_labels)

    # Calculate evaluation metrics for each task
    metrics = {}
    output_dir = "./classification/confusion_matrices/strategy2"  # Directory to save confusion matrices
    os.makedirs(output_dir, exist_ok=True)

    for task in ["gender", "bag", "hat"]:
        # Compute metrics using sklearn
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        # Print metrics for the task
        print(f"{task.capitalize()} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Compute and save the confusion matrix
        cm = confusion_matrix(all_labels[task], all_predictions[task])
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {task.capitalize()}")
        save_path = os.path.join(output_dir, f"confusion_matrix_{task}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix for {task} saved to {save_path}")

    return metrics


if __name__ == "__main__":
    # Test configuration
    data_dir = './dataset'  # Path to the dataset
    model_path = './models/classification_model_strategy2.pth'  # Path to the model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    # Define thresholds for each task
    thresholds = {"gender": 0.5, "bag": 0.5, "hat": 0.5}

    # Load the test dataset
    test_dataset = TestDataset(data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the pre-trained model
    model = PARMultiTaskNet(backbone='resnet50').to(device)
    checkpoint = torch.load(model_path, map_location=device)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state'])  # Restore the model state

    # Test the model
    print("\nTesting model...")
    metrics = test_model(model, test_loader, device, thresholds)

    # Print the summary of metrics for each task
    print("\nTest Summary:")
    for task, task_metrics in metrics.items():
        print(f"{task.capitalize()} Metrics: {task_metrics}")
