import os
from collections import Counter
import matplotlib.pyplot as plt
import random


def calculate_class_weights_from_file(file_path='./dataset/training_set.txt'):
    """
    Calculates the class weights based on the label frequencies by reading from the label file.

    Args:
        file_path (str): Path to the file containing the labels.

    Returns:
        dict: A dictionary with the weights for each class.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize a dictionary to store labels for each task
    labels = {'gender': [], 'bag': [], 'hat': []}

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                # Extract labels for gender, bag, and hat tasks
                labels['gender'].append(int(parts[3]))
                labels['bag'].append(int(parts[4]))
                labels['hat'].append(int(parts[5]))

    # Calculate weights for each task based on label frequencies
    class_weights = {}
    for task, task_labels in labels.items():
        label_counts = Counter(task_labels)
        max_count = max(label_counts.values())
        class_weights[task] = {label: max_count / count for label, count in label_counts.items() if label != -1}

    # Log the distribution and weights for each task
    for task, weights in class_weights.items():
        print(f"Label distribution for {task}: {Counter(labels[task])}")
        print(f"Class weights for {task}: {weights}")

    return class_weights, labels


def plot_label_distribution(labels, output_path='./classification/statistics/'):
    """
    Creates bar plots to visualize the label distribution for gender, bag, and hat tasks.

    Args:
        labels (dict): A dictionary with lists of labels for each task.
        output_path (str): Path to save the generated plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Generate a plot for each task
    for task, task_labels in labels.items():
        label_counts = Counter(task_labels)
        labels_, counts = zip(*label_counts.items())

        plt.figure(figsize=(8, 5))
        plt.bar(labels_, counts, color='skyblue')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'Label Distribution for {task}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot to the specified directory
        plt.savefig(os.path.join(output_path, f'{task}_distribution.png'))
        plt.close()
        print(f"Distribution plot for {task} saved in {output_path}")



def split_train_val(input_file, train_file, val_file, train_ratio=0.8, random_seed=None):
    """
    Splits a text file into two separate files: training and validation.

    Args:
        input_file (str): Path to the input file.
        train_file (str): Path to the output file for the training set.
        val_file (str): Path to the output file for the validation set.
        train_ratio (float): Proportion of data assigned to the training set (default: 0.8).
        random_seed (int, optional): Seed for the random number generator for reproducibility.
    """
    # Set the random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Read all lines from the input file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the split index
    split_index = int(len(lines) * train_ratio)

    # Divide the lines into training and validation sets
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    # Write the lines to the respective output files
    with open(train_file, 'w') as train_outfile:
        train_outfile.writelines(train_lines)

    with open(val_file, 'w') as val_outfile:
        val_outfile.writelines(val_lines)


if __name__ == "__main__":
    # Uncomment and modify the following lines for specific use cases:

    # Split the dataset into training and validation sets
    # input_file = "./dataset/training_set.txt"
    # train_file = "./dataset/train_split.txt"
    # val_file = "./dataset/val_split.txt"
    # split_train_val(input_file, train_file, val_file, 0.8, 65464)
    # print("Dataset split into training and validation sets.")

    #--- TRAIN PLOT ---
    weights, labels = calculate_class_weights_from_file("./dataset/train_split.txt")

    # --- VALIDATION PLOT ---
    #weights, labels = calculate_class_weights_from_file("./dataset/val_split.txt")

    # --- TEST PLOT ---
    #weights, labels = calculate_class_weights_from_file("./dataset/test_set.txt")

    # Print the calculated weights
    print("Calculated weights:", weights)

    # Generate and save the distribution plots
    plot_label_distribution(labels)
