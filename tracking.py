import cv2
from PIL import Image
from ultralytics import YOLO
from OutputWriter import OutputWriter
from classification.nets import PARMultiTaskNet
from lines_utils import get_lines_info, check_crossed_lines
import gui_utils as gui
import torch
import torchvision.transforms as T
from collections import defaultdict


def start_track(device, model_path="models/yolo11m.pt", video_path="videos/video.mp4", show=True, tracker="confs/botsort.yaml"):
    """
    Main function to perform people tracking in a video using a pre-trained YOLO model.
    At each frame, YOLO detects people and tracks their movement using unique IDs.
    The system performs classification for attributes such as gender, bag, and hat,
    and visualizes this information on the frame. Additionally, it calculates intersections
    with predefined virtual lines to detect crossings and updates the person's trajectory.

    Parameters:
    device: specifies the device to run the model on (e.g. 'cuda' for GPU or 'cpu')
    model_path: path to the YOLO model file
    video_path: path to the video to analyze
    show: flag to display the results in real time
    tracker: path to the tracker configuration file
    """

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Load the classification model for pedestrian attributes (gender, bag, hat)
    classification = PARMultiTaskNet(backbone='resnet50', pretrained=False, attention=True).to(device)
    checkpoint_path = './models/classification_model_strategy2.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classification.load_state_dict(checkpoint['model_state'])
    classification.eval()

    # Define image transformations for classification
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data structure initializations
    probability_sum_gender = defaultdict(float)  # Stores cumulative gender probabilities for each person ID
    probability_sum_bag = defaultdict(float)  # Stores cumulative bag probabilities for each person ID
    probability_sum_hat = defaultdict(float)  # Stores cumulative hat probabilities for each person ID
    number_of_inferences = {}  # Number of inferences made for each person ID
    pedestrian_attributes = {}  # Dict to store attributes (gender, bag, hat) of pedestrians
    track_history = defaultdict(lambda: [])  # Stores the movement history (trajectory) of each person by their ID
    crossed_line_by_people = defaultdict(lambda: [])  # Stores the lines that have been crossed by each person ID

    # Load lines information
    lines_info = get_lines_info()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Calculate the FPS of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 1)

    # Constant for processing frame rate and inference rate
    PROCESSING_FRAME_RATE = 10  # Number of frames to process per second
    FRAME_TO_SKIP = int(fps / PROCESSING_FRAME_RATE)  # Number of frames to skip to achieve the desired frame processing rate
    INFERENCE_RATE = FRAME_TO_SKIP * 2  # Define how often to perform classification (inferences)

    # Thresholds for classification
    GENDER_THRESHOLD = 0.5
    BAG_THRESHOLD = 0.5
    HAT_THRESHOLD = 0.75

    frame_count = 0  # Current frame

    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()

        # Skip frames to adjust the processing rate
        for i in range(FRAME_TO_SKIP):
            cap.grab()  # Grab frames without decoding them
        frame_count += FRAME_TO_SKIP

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker=tracker, classes=[0])  # classes=[0] is people
            if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                people_ids = results[0].boxes.id.int().cpu().tolist()

                # Create a copy of the frame for annotations
                annotated_frame = frame.copy()

                # Text with general scene information (total people, line crossings)
                text = [f"Total People: {len(people_ids)}"]
                for line in lines_info:
                    text.append(f"Passages for line {line['line_id']}: {line['crossing_counting']}")

                gui.add_info(annotated_frame, text)  # draw general information about the scene
                gui.add_lines_on_frame(annotated_frame, lines_info)  # draw the lines

                # For each bounding boxes and people_id
                for box, people_id in zip(boxes, people_ids):

                    # Initialize inference counter and pedestrian attributes for the person if not already present
                    if people_id not in number_of_inferences:
                        number_of_inferences[people_id] = 0
                        pedestrian_attributes[people_id] = "?"

                    x, y, w, h = box  # (x, y) = coordinates of the center; (w, h) = width and height of the bounding box
                    top_left_corner = (int(x - w / 2), int(y - h / 2))  # top-left corner of the bounding box
                    bottom_right_corner = (int(x + w / 2), int(y + h / 2))  # bottom-right corner of the bounding box
                    bottom_left_corner = (int(x - w / 2), int(y + h / 2))  # bottom-left corner of the bounding box

                    gui.add_bounding_box(annotated_frame, top_left_corner, bottom_right_corner)  # draw bounding box
                    gui.add_people_id(annotated_frame, people_id, top_left_corner)  # draw people id

                    # Track of the lower center of the bounding box in the last two frame
                    track = track_history[people_id]
                    track.append((float(x), float(y + h / 2)))  # lower center of the bounding box
                    if len(track) > 2:  # Keep track of only the last two points
                        track.pop(0)

                    # Check if the person crossed any lines
                    crossed_line_ids = check_crossed_lines(track, lines_info)  # returns the lines crossed by the person
                    crossed_line_by_people[people_id].extend(crossed_line_ids)

                    # Inference for pedestrian attributes (gender, bag, hat)
                    if frame_count % INFERENCE_RATE == 0:
                        # extract the image of the person and apply the transformations to it
                        people_img = gui.get_bounding_box_image(frame, top_left_corner, bottom_right_corner)
                        people_img = transforms(Image.fromarray(people_img).convert('RGB')).unsqueeze(0).to(device)

                        # Perform classification (inference)
                        outputs = classification(people_img)
                        number_of_inferences[people_id] += 1

                        # Update cumulative probabilities for each attribute
                        probability_sum_gender[people_id] += torch.sigmoid(outputs["gender"]).item()
                        probability_sum_bag[people_id] += torch.sigmoid(outputs["bag"]).item()
                        probability_sum_hat[people_id] += torch.sigmoid(outputs["hat"]).item()

                        # Calculate the averages of the classifications made up to this point in order to classify the attributes
                        gender = 1 if (probability_sum_gender[people_id] / number_of_inferences[people_id]) > GENDER_THRESHOLD else 0
                        bag = 1 if (probability_sum_bag[people_id] / number_of_inferences[people_id]) > BAG_THRESHOLD else 0
                        hat = 1 if (probability_sum_hat[people_id] / number_of_inferences[people_id]) > HAT_THRESHOLD else 0

                        # Create the list of pedestrian attributes and trajectory to display
                        pedestrian_attributes[people_id] = [f"Gender: {'F' if gender else 'M'}"]
                        if bag and hat:
                            pedestrian_attributes[people_id].append("Bag Hat")
                        elif bag:
                            pedestrian_attributes[people_id].append("Bag")
                        elif hat:
                            pedestrian_attributes[people_id].append("Hat")
                        else:
                            pedestrian_attributes[people_id].append("No Bag No Hat")
                        pedestrian_attributes[people_id].append(f"{crossed_line_by_people.get(people_id)}")

                    # Draw the pedestrian attributes below the bounding box
                    gui.add_info(annotated_frame, pedestrian_attributes[people_id], bottom_left_corner, 0.5, 2)

                # Display the annotated frame
                if show:
                    cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("YOLO Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("YOLO Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    # Write results to a file
    output_writer = OutputWriter()
    for people_id in number_of_inferences:
        if number_of_inferences[people_id] != 0:  # for each classified person
            gender = 1 if (probability_sum_gender[people_id] / number_of_inferences[people_id]) > GENDER_THRESHOLD else 0
            bag = 1 if (probability_sum_bag[people_id] / number_of_inferences[people_id]) > BAG_THRESHOLD else 0
            hat = 1 if (probability_sum_hat[people_id] / number_of_inferences[people_id]) > HAT_THRESHOLD else 0
            trajectory = crossed_line_by_people[people_id]
            output_writer.add_person(people_id, gender, bag, hat, trajectory)
    output_writer.write_output()