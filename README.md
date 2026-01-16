<h1> Artificial Vision Project 2024/2025 </h1>

<h2> Project Description </h2>
This project, part of the Artificial Vision Contest 2025, aims to develop an advanced system for automatic video analysis.
The proposed software will:
Detect and track all people in real-time within a video.
Recognize pedestrian attributes such as gender, bags, hats.
Analyze behaviors by monitoring virtual line crossings.

<h2> Goals </h2>
Ensure high precision in detection and classification to minimize errors. The system will process videos in real-time, display results through an interactive interface, and generate a standardized output file.

<h2> Project Structure </h2> 

```plaintext

├── classification/                                # Classification files
│   ├── dataset.py                                 # Dataset handling for classification tasks
│   ├── nets.py                                    # Model definitions and architectures for pedestrian attribute classification
│   ├── preprocess.py                              # Data preprocessing pipeline
│   ├── train_strategy1.py                         # Training script for strategy 1
│   ├── train_strategy2.py                         # Training script for strategy 2
│   └── test.py                                    # Testing and evaluation script
│
├── confs/                                         # Configuration files
│   ├── botsort.yaml                               # Tracker configuration for BoT-SORT
│   └── config.txt                                 # Configuration files for camera and lines
│
├── dataset/                                       # Dataset-related files
│
├── models/                                        # Models directory
│   ├── yolo11m.pt                                 # Pre-trained YOLO model for pedestrian detection
│   ├── classification_model_strategy1.pth         # Trained model for classification with strategy 1
│   └── classification_model_strategy2.pth         # Trained model for classification with strategy 2
│
├── result/                                        # Result files
│   └── result.txt                                 # Result file to store analyzed results
│
├── videos/                                        # Video files for testing and analysis
│
├── gui_utils.py                                   # GUI utilities for rendering bounding boxes, text, and other visual elements
├── lines_utils.py                                 # Utilities for managing and checking line crossings
├── main.py                                        # Main script for running the pedestrian analysis system
├── OutputWriter.py                                # Manages saving results and generating reports
├── tracking.py                                    # Tracking-related functions using BoT-SORT
└── README.md                                      # Project documentation

```

<h2> Requirements </h2>
Programming Language : Python 3.+

<h2> Dependencies </h2>
torch
torchvision
ultralytics
opencv-python
Pillow
numpy
matplotlib
scikit-learn
tqdm
seaborn

<h2> Recommended Hardware </h2>
GPU with CUDA support for optimal performance.

<h2> Execution </h2>
- To run the project on the example video "videos/Atrio.mp4", execute the main file:
  python main.py
  
- Full datasets are avaiable at:
    Training images
    https://drive.google.com/file/d/1uEcO7zgZilzDhbr1wdGkoG6LxH_uJ8ay/view?usp=share_link
    Validation images
    https://drive.google.com/file/d/1HXJdXgnjYb2AcHO841McnUlw4roJthK-/view?usp=share_link
  
- Trained models are not avaiable
