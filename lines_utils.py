import numpy as np
from math import cos, sin
import json


def mapping_3D_2D(x_real, y_real, config_file="confs/config.txt"):
    """
    Converts real-world coordinates to pixel coordinates on the image plane.
    This function applies a series of transformations (including yaw, pitch, and roll)
    to project the real-world coordinates in 3D space onto pixel coordinates in the 2D
    image plane, using camera parameters loaded from a configuration file.

    Parameters:
    x_real : numpy.ndarray
        Vector containing the real-world x coordinates of the points to be projected
    y_real : numpy.ndarray
        Vector containing the real-world y coordinates of the points to be projected
    config_file : str, optional
        Path to the configuration text file (optional, default is "confs/config.txt")

    Returns:
    u : numpy.ndarray
        Vector of x coordinates in pixels on the image plane
    v : numpy.ndarray
        Vector of y coordinates in pixels on the image plane
    """
    # Load camera parameters from the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Camera parameters from the configuration file
    f = config['f']  # Focal length in meters
    U = config['U']  # Image width in pixels
    V = config['V']  # Image height in pixels
    yaw = config['thyaw']  # Yaw rotation in radians
    roll = config['throll']  # Roll rotation in radians
    pitch = config['thpitch']  # Pitch rotation in radians
    xc, yc, zc = config['xc'], config['yc'], config['zc']  # Camera position in the world coordinate system
    s_w = config['sw']  # Sensor width
    s_h = config['sh']  # Sensor height

    # Transform real-world coordinates into the camera's coordinate system
    real_coordinates = np.vstack((x_real, y_real, np.zeros_like(x_real)))  # Add z=0 for the points
    camera_position = np.array([xc, yc, zc]).reshape(3, 1)
    translated_coordinates = real_coordinates - camera_position  # Translation to move the point to the camera's origin

    # Rotation matrices
    R_yaw = np.array([[cos(yaw), sin(yaw), 0],
                      [-sin(yaw), cos(yaw), 0],
                      [0, 0, 1]])

    R_roll = np.array([[cos(roll), 0, -sin(roll)],
                       [0, 1, 0],
                       [sin(roll), 0, cos(roll)]])

    R_pitch = np.array([[1, 0, 0],
                        [0, cos(pitch), sin(pitch)],
                        [0, -sin(pitch), cos(pitch)]])

    # Total rotation matrix
    R = R_roll @ R_pitch @ R_yaw

    # Apply the rotation: transform into the camera's coordinate system
    camera_coordinates = R @ translated_coordinates

    # Extract x, y, z coordinates in the camera's coordinate system
    dx = camera_coordinates[0, :]  # x coordinates in the camera's coordinate system
    dy = camera_coordinates[1, :]  # y coordinates in the camera's coordinate system
    dz = camera_coordinates[2, :]  # z coordinates in the camera's coordinate system

    # Compute the focal length in pixels
    f_x = f / s_w * U
    f_y = f / s_h * V

    # Projection: calculate the pixel coordinates (u, v)
    u = U / 2 + f_x * dx / dy
    v = V / 2 - (f_x * dz / dy)

    # Round the coordinates to integers
    for i in range(len(u)):
        u[i], v[i] = int(u[i]), int(v[i])

    # Error handling: check if the projected coordinates are within the image resolution
    if np.any(u < 0) or np.any(u > U) or np.any(v < 0) or np.any(v > V):
        raise ValueError(f"Pixel coordinates (u, v) are out of image resolution: u={u}, v={v}")

    return u, v


def load_lines(config_file="confs/config.txt"):
    """
    This function loads the lines from a text file formatted in JSON, extracts their real-world coordinates
    and then uses the `mapping_3D_2D` function to obtain pixel coordinates in the image.

    Parameters:
    config_file : str
        Path to the configuration text file (optional, default is "confs/config.txt")

    Returns:
    lines_in_pixel : list
        A list of dictionaries containing lines with pixel coordinates (x1, y1, x2, y2)
    """
    # Load parameters from the configuration text file
    with open(config_file, 'r') as f:
        config = json.load(f)  # Assuming the file contains JSON-formatted content

    # Lists to store the lines in real-world coordinates and pixel coordinates
    lines_in_pixel = []

    # Iterate over the lines in the configuration file
    for line in config['lines']:
        # Extract the real-world coordinates of the lines (x1, y1, x2, y2)
        id, x1_real, y1_real, x2_real, y2_real = line['id'], line['x1'], line['y1'], line['x2'], line['y2']

        try:
            # Convert the real-world coordinates to pixel coordinates
            x_pixel, y_pixel = mapping_3D_2D(np.array([x1_real, x2_real]), np.array([y1_real, y2_real]))

            # Save the lines with pixel coordinates in the `lines_in_pixel` list
            lines_in_pixel.append({
                'id': id,  # Line ID
                'x1': x_pixel[0],  # x-coordinate of the first point in pixels
                'y1': y_pixel[0],  # y-coordinate of the first point in pixels
                'x2': x_pixel[1],  # x-coordinate of the second point in pixels
                'y2': y_pixel[1]  # y-coordinate of the second point in pixels
            })
        except ValueError as e:
            print(e)

    return lines_in_pixel


def get_lines_info():
    """
    This function processes the lines, extracts their IDs and coordinates, calculates the text position
    for labeling and computes the direction for the arrow placement.

    Returns:
    info_lines : list
        A list of dictionaries, each containing the following keys:
        - 'line_id': ID of the line
        - 'start_point': starting point of the line
        - 'end_point': ending point of the line
        - 'id_position': coordinates for the id position
        - 'start_arrow': starting point of the arrow
        - 'end_arrow': ending point of the arrow
        - 'crossing_counting': the number of times the line has been crossed
    """
    # Load the lines from the configuration file
    lines_in_pixel = load_lines()

    # Initialize empty lists to store the line IDs and points
    ids = []
    points = []

    # Iterate through the lines in the input list
    for line in lines_in_pixel:
        # Add the line ID to the 'ids' list
        ids.append(line['id'])

        # Add the start and end points of the line to the 'points' list
        points.append((line['x1'], line['y1']))  # First point
        points.append((line['x2'], line['y2']))  # Second point

    info_lines = []  # List to store the detailed information of the lines
    crossing_counting = 0  # Initialize the crossing counter

    # Process each line to generate detailed information
    for i in range(len(ids)):
        line_id = ids[i]

        # Get the start and end points for the current line
        j = i * 2  # Each ID corresponds to two points (start and end)
        start_point = points[j]
        end_point = points[j + 1]

        # Convert points to integers
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))

        # Calculate the text position slightly above the start of the line
        if start_point[0] < end_point[0]:
            id_position = (start_point[0] - 15, start_point[1] - 15)
        else:
            id_position = (end_point[0] - 15, end_point[1] - 15)

        # Calculate the vector pointing from the start_point to the end_point
        dx = start_point[0] - end_point[0]
        dy = start_point[1] - end_point[1]

        # Euclidean distance between the start and end points
        length = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate the normalized components of the vector
        unit_dx = dx / length
        unit_dy = dy / length

        # Perpendicular direction (rotation of vector by 90 degrees)
        perp_dx = -unit_dy
        perp_dy = unit_dx

        # Calculate the midpoint of the line
        mid_x = (start_point[0] + end_point[0]) // 2
        mid_y = (start_point[1] + end_point[1]) // 2

        # Calculate the arrow position
        arrow_length = 50  # Length of the arrow
        start_arrow = (int(mid_x), int(mid_y))
        end_arrow = (int(mid_x + perp_dx * arrow_length), int(mid_y + perp_dy * arrow_length))

        # Append the line's detailed information to the list
        info_lines.append({
            'line_id': line_id,  # ID of the line
            'start_point': start_point,  # Starting point of the line
            'end_point': end_point,  # Ending point of the line
            'id_position': id_position,  # Position for the text label
            'start_arrow': start_arrow,  # Starting point of the arrow
            'end_arrow': end_arrow,  # Ending point of the arrow
            'crossing_counting': crossing_counting  # Number of times the line has been crossed (initially 0)
        })

    return info_lines


def orientation(p, q, r):
    """
    This function calculates the orientation of the triplet of points (p, q, r).
    The orientation is determined by the cross product of the vectors pq and pr.
    - If the result is positive, the points are in counter-clockwise order
    - If the result is negative, the points are in clockwise order
    - If the result is zero, the points are collinear

    Parameters:
    p : tuple
        The first point (x1, y1)
    q : tuple
        The second point (x2, y2)
    r : tuple
        The third point (x3, y3)

    Returns:
    float
        The orientation value (positive, negative, or zero)
    """
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def on_segment(p, q, r):
    """
    This function checks if point q lies on the segment pr.

    Parameters:
    p : tuple
        The first point of the segment (x1, y1)
    q : tuple
        The point to check (x2, y2)
    r : tuple
        The second point of the segment (x3, y3)

    Returns:
    bool
        True if point q lies on the segment pr, False otherwise
    """
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def intersects(p1, p2, p3, p4):
    """
    This function determines whether the segment p1p2 intersects with the segment p3p4.
    The function checks the orientations of the triplets (p1, p2, p3), (p1, p2, p4),
    (p3, p4, p1), and (p3, p4, p2) to determine if the segments intersect.
    It also handles collinear cases.

    Parameters:
    p1 : tuple
        The first point of the first segment (x1, y1)
    p2 : tuple
        The second point of the first segment (x2, y2)
    p3 : tuple
        The first point of the second segment (x3, y3)
    p4 : tuple
        The second point of the second segment (x4, y4)

    Returns:
    bool
        True if the segments intersect, False otherwise
    """
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # Check if the segments properly intersect
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Check for collinear cases
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


def check_crossed_lines(track, lines_info):
    """
    This function checks which lines in the lines_info list are crossed by the given track.
    It checks if the last two points of the track intersect any of the lines and if the
    direction of the track is in the same direction as the arrow on the line.

    Parameters:
    track : list
        A list of points representing the track (each point is a tuple (x, y))
    lines_info : list
        A list of dictionaries representing the lines, where each dictionary contains:
        - 'line_id': the ID of the line
        - 'start_point': the start point of the line (x1, y1)
        - 'end_point': the end point of the line (x2, y2)
        - 'start_arrow': the start point of the arrow (x3, y3)
        - 'end_arrow': the end point of the arrow (x4, y4)
        - 'crossing_counting': the count of times the line has been crossed

    Returns:
    list
        A list of the IDs of the lines that have been crossed
    """
    crossed_line_ids = []  # List to store the IDs of crossed lines

    for line in lines_info:
        # Extract line information
        line_id = line['line_id']
        start_point = line['start_point']
        end_point = line['end_point']
        start_arrow = line['start_arrow']
        end_arrow = line['end_arrow']

        # Get the last two points of the track (representing the track's movement)
        end_track = track[len(track) - 1]
        start_track = track[len(track) - 2]

        # Check if the track segment intersects with the line
        if intersects(start_track, end_track, start_point, end_point):
            # Calculate the vectors for the track and the line's arrow
            track_vector = np.array([end_track[0] - start_track[0], end_track[1] - start_track[1]])
            arrow_vector = np.array([end_arrow[0] - start_arrow[0], end_arrow[1] - start_arrow[1]])

            # Calculate the dot product to check if the track direction matches the arrow direction
            dot_product = np.dot(track_vector, arrow_vector)

            # If the dot product is positive, the track is moving in the same direction as the arrow
            if dot_product > 0:
                line['crossing_counting'] += 1  # Increment the crossing count
                crossed_line_ids.append(line_id)  # Add the line ID to the list of crossed lines

    return crossed_line_ids
