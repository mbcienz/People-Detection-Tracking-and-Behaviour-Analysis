import os
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import TrainDataset, ValidationDataset
from nets import PARMultiTaskNet
import torch
import numpy as np


def calculate_class_weights(dataset):
    """
    Calculate class weights for each task to balance the dataset and assign a weight to each sample.

    :param dataset: PyTorch dataset
    :return: Array of weights for each sample
    """
    # Precomputed class distributions with a fixed seed=65464
    gender_dist = Counter({0: 49386, 1: 18968, -1: 6110})
    bag_dist = Counter({0: 44246, -1: 21773, 1: 8445})
    hat_dist = Counter({0: 54983, -1: 11794, 1: 7687})

    scale_factor = 1000
    gender_weights = {label: (1.0 / count) * scale_factor for label, count in gender_dist.items() if label != -1}
    bag_weights = {label: (1.0 / count) * scale_factor for label, count in bag_dist.items() if label != -1}
    hat_weights = {label: (1.0 / count) * scale_factor for label, count in hat_dist.items() if label != -1}

    sample_weights = []
    for i in range(len(dataset)):

        # Extract sample labels
        labels = np.array(dataset[i][1])

        # Compute weights for each task, assigning 0.0 if the label is -1
        gender_weight = gender_weights.get(labels[0].item(), 0.0)
        bag_weight = bag_weights.get(labels[1].item(), 0.0)
        hat_weight = hat_weights.get(labels[2].item(), 0.0)

        # Assign 0.0 weight if all labels are -1
        if all(label == -1 for label in labels):
            combined_weight = 0.0
        else:
            # Calculate the combined weight as the mean of valid weights
            combined_weight = np.mean([gender_weight, bag_weight, hat_weight])

        sample_weights.append(combined_weight)

    return np.array(sample_weights)



def initialize_weights(module):
    """
    Initialize model weights using Xavier Uniform initialization.

    :param module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def masked_loss(criterion, outputs, labels, mask):
    """
    Apply masked loss, excluding contributions from labels equal to -1.

    :param criterion: Loss function (e.g., BCEWithLogitsLoss)
    :param outputs: Model outputs
    :param labels: Target labels
    :param mask: Binary mask (1 where label >= 0, 0 otherwise)
    :return: Masked loss
    """
    masked_outputs = outputs[mask]
    masked_labels = labels[mask]
    if masked_outputs.numel() == 0:  # Avoid error if the mask is empty
        return torch.tensor(0.0, device=outputs.device).mean()
    return criterion(masked_outputs, masked_labels).mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train the model for one epoch.

    :param model: Multi-task model
    :param dataloader: Training data loader
    :param optimizer: Optimizer
    :param device: Device (e.g., 'cuda' or 'cpu')
    :param epoch: Current epoch
    :return: Average loss for the epoch
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch

    # Iterate through the dataloader
    for images, labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)  # Move data to the specified device
        masks = labels >= 0  # Create masks for valid labels (>= 0)
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = model(images)  # Forward pass

        # Compute losses for each task
        gender_loss = masked_loss(model.gender_loss, outputs["gender"].squeeze(), labels[:, 0], masks[:, 0])
        bag_loss = masked_loss(model.bag_loss, outputs["bag"].squeeze(), labels[:, 1], masks[:, 1])
        hat_loss = masked_loss(model.hat_loss, outputs["hat"].squeeze(), labels[:, 2], masks[:, 2])

        # Combine losses equally across tasks
        loss = 1/3 * (gender_loss + bag_loss + hat_loss)

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate the loss

    # Return the average loss over all batche
    return running_loss / len(dataloader)


def validate(model, dataloader, device, epoch):
    """
    Validate the model on the validation dataset.

    :param model: Multi-task model
    :param dataloader: Validation data loader
    :param device: Device (e.g., 'cuda' or 'cpu')
    :param epoch: Current epoch
    :return: Tuple of average loss and metrics dictionary
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0  # Initialize running loss for the epoch

    # Initialize containers for labels and predictions for each task
    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    # Define thresholds for each task
    thresholds = {"gender": 0.5, "bag": 0.5, "hat": 0.5}

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            # Create masks for valid labels (>= 0)
            masks = labels >= 0

            outputs = model(images)  # Forward pass

            # Compute masked losses for each task
            gender_loss = masked_loss(model.gender_loss, outputs["gender"].squeeze(), labels[:, 0], masks[:, 0])
            bag_loss = masked_loss(model.bag_loss, outputs["bag"].squeeze(), labels[:, 1], masks[:, 1])
            hat_loss = masked_loss(model.hat_loss, outputs["hat"].squeeze(), labels[:, 2], masks[:, 2])

            # Combine losses equally across tasks
            loss = 1 / 3 * (gender_loss + bag_loss + hat_loss)

            running_loss += loss.item()  # Accumulate the loss

            # Store predictions and labels for each task
            for task in ["gender", "bag", "hat"]:

                # Get the threshold for the current task
                threshold = thresholds[task]

                preds = (torch.sigmoid(outputs[task]) > threshold).int()  # Apply sigmoid and threshold
                task_index = ["gender", "bag", "hat"].index(task)  # Get the index of the current task
                all_predictions[task].extend(preds[masks[:, task_index]].cpu().numpy())
                all_labels[task].extend(labels[masks[:, task_index], task_index].cpu().numpy())

    # Calculate metrics for each task
    metrics = {}
    for task in ["gender", "bag", "hat"]:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Return the average loss and computed metrics
    return running_loss / len(dataloader), metrics


def plot_metrics(metrics_history, output_dir, epoch, loss_history, val_loss_history):
    """
    Plot and save the metrics and loss curves for training and validation.

    :param metrics_history: List of metrics dictionaries for each epoch
    :param output_dir: Directory to save the plots
    :param epoch: Current epoch
    :param loss_history: List of training loss values for each epoch
    :param val_loss_history: List of validation loss values for each epoch
    """
    # Plot metrics for each task
    for task in metrics_history[0].keys():
        plt.figure()
        for metric in ["accuracy", "precision", "recall", "f1"]:
            plt.plot(
                [epoch_metrics[task][metric] for epoch_metrics in metrics_history], label=f"{task}_{metric}"
            )
        plt.title(f"{task.capitalize()} Metrics (up to Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{task}_metrics.png"))
        plt.close()

    # Plot training loss
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    # Plot validation loss
    plt.figure()
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Loss")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "val_loss.png"))
    plt.close()



def main():
    """
    Main function to train and validate the multi-task model.
    Parses arguments, initializes datasets, model, optimizer, and scheduler,
    and performs training and validation for the specified number of epochs.
    """
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda o cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./classification/checkpoints',
                        help='Directory dei checkpoint')
    parser.add_argument('--resume_checkpoint', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone name: resnet50 o resnet18')
    parser.add_argument('--balancing', type=bool, default=True, help='Balancing batches')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--optimizer', type=str, default="sgd", help='adam o sgd')
    parser.add_argument('--attention', type=bool, default=True, help='use CBAM attention')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure the checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load training and validation datasets
    print("Initializing training set...")
    train_dataset = TrainDataset(data_dir=args.data_dir)
    print("Initializing validation set...")
    val_dataset = ValidationDataset(data_dir=args.data_dir)

    if args.balancing:
        print("Calculating class weights for batch balancing...")
        class_weights = calculate_class_weights(train_dataset)
        sampler = WeightedRandomSampler(class_weights, len(train_dataset))
    else:
        sampler = SubsetRandomSampler(list(range(len(train_dataset))))

    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    print("Initializing model...")
    model = PARMultiTaskNet(backbone=args.backbone, pretrained=True, attention=args.attention).to(device)

    # Initialize only the task-specific heads
    model.gender_head.apply(initialize_weights)
    model.bag_head.apply(initialize_weights)
    model.hat_head.apply(initialize_weights)

    # Configure optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            [
                {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": model.gender_head.parameters(), "lr": args.lr},
                {"params": model.bag_head.parameters(), "lr": args.lr},
                {"params": model.hat_head.parameters(), "lr": args.lr}
            ]
        )
    else:
        optimizer = optim.SGD([
            {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
            {'params': model.gender_head.parameters()},
            {'params': model.bag_head.parameters()},
            {'params': model.hat_head.parameters()}
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Initialize variables for tracking training
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = []
    loss_history = []
    val_loss_history = []

    # Resume training from a checkpoint if specified
    if args.resume_checkpoint:
        checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda f: os.path.getctime(os.path.join(args.checkpoint_dir, f))
            )
            checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
            print(f"Loading model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if checkpoint['scheduler_state'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
        else:
            print("No checkpoint found. Starting from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    # Begin training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate the model
        val_loss, val_metrics = validate(model, val_loader, device, epoch)
        metrics_history.append(val_metrics)
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Print training and validation results
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for task, metrics in val_metrics.items():
            print(
                f"{task.capitalize()} - Validation Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f},"
                f" Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        # Plot and save metrics
        plot_metrics(metrics_history, args.checkpoint_dir, epoch + 1, loss_history, val_loss_history)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        # Trigger early stopping if patience is exceeded
        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
