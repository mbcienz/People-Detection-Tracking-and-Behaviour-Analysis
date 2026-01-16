import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class PARCustomDataset(Dataset):
    def __init__(self, data_dir="./dataset", txt_file="training_set.txt", transforms=None):
        """
        Base class for handling the PAR (Pedestrian Attribute Recognition) dataset.

        :param data_dir: Path to the directory containing images (default: "./dataset")
        :param txt_file: Text file containing image annotations (default: "training_set.txt")
        :param transforms: Transformations to be applied to the images
        """
        self.data_dir = data_dir
        self.txt_file = txt_file
        self.transforms = transforms

        # Load data from the annotation file
        self.data = []
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            for line in f:
                try:
                    # Parse each line of the annotation file
                    parts = line.strip().split(',')
                    img_name, gender, bag, hat = parts[0], int(parts[3]), int(parts[4]), int(parts[5])
                    img_subdir = "test_set" if "test_set" in txt_file else "training_set"
                    img_path = os.path.join(data_dir, img_subdir, img_name)

                    if os.path.exists(img_path):

                        # Avoid labels = -1, -1, -1
                        if gender != -1 or bag != -1 or hat != -1:
                            self.data.append((img_path, [gender, bag, hat]))
                    else:
                        print(f"Missing image: {img_path}")
                except Exception as e:
                    print(f"Error parsing line: {line}. Details: {e}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Fetches an item from the dataset at a specific index.

        :param index: Index of the desired item
        :return: Tuple (image, label)
        """
        try:
            img_path, labels = self.data[index]
            image = Image.open(img_path).convert('RGB')  # Open and convert the image to RGB format

            # Apply transformations if specified
            if self.transforms:
                image = self.transforms(image)
            return image, torch.tensor(labels, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image or labels at index {index}. Details: {e}")
            return None, None


class TrainDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="train_split.txt"):
        """
        Subclass for handling the training dataset

        :param data_dir: Path to the directory containing images (default: "./dataset")
        :param txt_file: Annotation file for training (default: "train_split.txt")
        """
        # Specify transformations for training
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(data_dir, txt_file, transforms)


class ValidationDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="val_split.txt"):
        """
        Initializes the validation dataset.

        :param data_dir: Path to the directory containing images (default: "./dataset")
        :param txt_file: Annotation file for validation (default: "val_split.txt")
        """
        # Specify transformations for validation
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(data_dir, txt_file, transforms)


class TestDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="test_set.txt"):
        """
        Initializes the test dataset.

        :param data_dir: Path to the directory containing images (default: "./dataset")
        :param txt_file: Annotation file for testing (default: "test_set.txt")
        """
        # Specify transformations for testing
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        super().__init__(data_dir, txt_file, transforms)


if __name__ == "__main__":
    # Test the dataset classes
    data_dir = "./dataset"

    # Instantiate the datasets
    train_dataset = TrainDataset(data_dir=data_dir)
    val_dataset = ValidationDataset(data_dir=data_dir)
    test_dataset = TestDataset(data_dir=data_dir)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Example
    for i in range(len(train_dataset)):
        image, labels = train_dataset[i]
        if image is not None and labels is not None:
            print(f"Image shape: {image.shape}, Labels: {labels}")
