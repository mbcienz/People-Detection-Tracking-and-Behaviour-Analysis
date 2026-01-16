import torch
from tracking import start_track


def main():
    """
    Main function to initialize tracking process on a video file.
    This function checks if CUDA (GPU support) is available and selects the appropriate device (CPU or GPU).
    Then, it starts tracking on the video specified in the `video_path`.
    """
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Path to the YOLO model file
    model_path = "models/yolo11m.pt"

    # Path to the video to analyze
    video_path = 'videos/Contest.mp4'

    # Flag to control if the processed video should be displayed
    show = True

    # Path to the tracker configuration file
    tracker = "confs/botsort.yaml"

    # Start the tracking process
    start_track(device, model_path, video_path, show, tracker)


if __name__ == "__main__":
    main()
