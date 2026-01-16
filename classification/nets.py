import torch.nn as nn
import torchvision.models as models
from ultralytics.nn.modules import CBAM
import torch.nn.functional as F


# Backbone network for feature extraction
class Backbone(nn.Module):
    """
    Initializes the backbone network for feature extraction.

    :param name: Backbone architecture name ("densenet", "resnet18", or "resnet50").
    :param pretrained: If True, uses a pretrained model on ImageNet.
    """

    def __init__(self, name="resnet50", pretrained=True):
        super(Backbone, self).__init__()

        if name == "densenet":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.classifier.in_features  # Number of channels in output

            # Remove the global average pooling e classifier layers
            self.backbone = nn.Sequential(*list(backbone.features.children()))
        elif name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.fc.in_features  # Number of channels in output

            # Remove the global average pooling e classifier layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.fc.in_features  # Number of channels in output

            # Remove the global average pooling e classifier layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Ensure all parameters have requires_grad=True
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.params = list(self.backbone.parameters())

    def forward(self, x):
        """
        Forward pass through the backbone.

        :param x: Input image tensor.
        :return: Extracted feature maps.
        """
        return self.backbone(x)


# Squeeze-and-Excitation (SE) block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Initializes the SE block.

        :param channels: Number of input channels.
        :param reduction: Reduction ratio for the bottleneck layer.
        """
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the SE block.

        :param x: Input tensor.
        :return: Channel-weighted output tensor.
        """
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Modulate the channels


# Classification head for multi-task learning
class ClassificationHead(nn.Module):
    """
    Initializes a classification head for an individual task.

    :param num_classes: Number of output classes.
    :param input_features: Number of input features from the backbone.
    :param attention: If True, applies attention mechanisms (CBAM and SEBlock).
    """
    def __init__(self, num_classes=1, input_features=2048, attention=True):
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.input_features = input_features
        self.attention = attention

        if self.attention:
            self.attention = CBAM(self.input_features)
            self.se_block = SEBlock(self.input_features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(self.input_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass through the classification head.

        :param x: Input tensor.
        :return: Class logits.
        """
        if self.attention:
            x = self.attention(x)
            x = self.se_block(x)  # SE Block after CBAM

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class PARMultiTaskNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, attention=True):
        """
        Initializes the multi-task network for pedestrian attribute recognition.

        :param backbone: Backbone architecture ("resnet50", "resnet18", "densenet").
        :param pretrained: If True, uses a pretrained backbone.
        :param attention: If True, enables attention mechanisms.
        """
        super(PARMultiTaskNet, self).__init__()

        # Defining backbone for feature extraction
        self.backbone = Backbone(name=backbone, pretrained=pretrained)

        # Heads for each task: gender, bag, and hat prediction
        self.gender_head = ClassificationHead(1, self.backbone.out_features, attention=attention)
        self.bag_head = ClassificationHead(1, self.backbone.out_features, attention=attention)
        self.hat_head = ClassificationHead(1, self.backbone.out_features, attention=attention)

        # Define individual binary cross-entropy loss for each task
        self.gender_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bag_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.hat_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        """
        Forward pass through the multi-task network.

        :param x: Input image tensor.
        :return: Dictionary containing predictions for each task.
        """
        features = self.backbone(x)

        gender_output = self.gender_head(features)
        bag_output = self.bag_head(features)
        hat_output = self.hat_head(features)

        return {
            "gender": gender_output,
            "bag": bag_output,
            "hat": hat_output
        }
