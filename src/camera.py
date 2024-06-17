import os
import cv2
import numpy as np
import time

def apply_gaussian_blur(frame, top_left_y, top_left_x, bottom_right_y, bottom_right_x):
    # Apply a blur to the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 100)

    # Copy the central region from the original frame to the blurred frame
    blurred_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return blurred_frame

def draw_rectangle(canvas, rect_top_left, rect_bottom_right, color=(255, 0, 0), thickness=2):
    """
    Draws a rectangle on the given canvas.

    Parameters:
    - canvas: The image canvas on which to draw the rectangle.
    - rect_top_left: Tuple (x, y) for the top-left corner of the rectangle.
    - rect_bottom_right: Tuple (x, y) for the bottom-right corner of the rectangle.
    - color: The color of the rectangle (default is blue).
    - thickness: The thickness of the rectangle border (default is 2).
    """
    cv2.rectangle(canvas, rect_top_left, rect_bottom_right, color, thickness)

def show_camera_and_return_frames(camera_index, target_fps, effective_height, effective_width, apply_blur, show_rectangle, flip_horizontal):
    # Open the default camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set the desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Get the maximum FPS of the camera
    max_fps = cap.get(cv2.CAP_PROP_FPS)
    if max_fps == 0:
        max_fps = 30  # Default to 30 if the information is not available

    # Calculate the interval for the target FPS
    target_interval = 1.0 / target_fps

    last_return_time = 0

    # Create a named window with the option to resize
    cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Get the current window size
        window_width = cv2.getWindowImageRect('Camera Stream')[2]
        window_height = cv2.getWindowImageRect('Camera Stream')[3]

        # Calculate the aspect ratio of the frame
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        # Calculate new dimensions to fit the window while maintaining aspect ratio
        if window_width / window_height > aspect_ratio:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black canvas
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Calculate padding
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2

        # Place the resized frame on the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

        # Calculate the coordinates of the central region
        center_y = frame_height // 2
        center_x = frame_width // 2
        top_left_y = max(center_y - effective_height // 2, 0)
        top_left_x = max(center_x - effective_width // 2, 0)
        bottom_right_y = min(center_y + effective_height // 2, frame_height)
        bottom_right_x = min(center_x + effective_width // 2, frame_width)

        effective_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        if apply_blur:
            blurred_frame = apply_gaussian_blur(frame, top_left_y, top_left_x, bottom_right_y, bottom_right_x)
            # Resize the blurred frame to fit the window
            resized_blurred_frame = cv2.resize(blurred_frame, (new_width, new_height))
            # Place the resized blurred frame on the canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_blurred_frame

        if show_rectangle:
            # Draw a blue rectangle around the effective central region
            rect_top_left = (x_offset + (new_width * top_left_x) // frame_width, y_offset + (new_height * top_left_y) // frame_height)
            rect_bottom_right = (x_offset + (new_width * bottom_right_x) // frame_width, y_offset + (new_height * bottom_right_y) // frame_height)
            draw_rectangle(canvas, rect_top_left, rect_bottom_right)

        # Display the resulting frame
        cv2.imshow('Camera Stream', canvas)

        # Get the current time
        current_time = time.time()

        # Check if it's time to return a frame
        if current_time - last_return_time >= target_interval:
            last_return_time = current_time
            yield effective_frame  # Return the effective frame at the specified FPS

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_index = 0
    target_fps = 1  # Define the target FPS for returning frames
    effective_height = 1000  # Define the effective height of the central region
    effective_width = 1000  # Define the effective width of the central region
    apply_blur = True  # Toggle Gaussian blur on or off
    show_rectangle = True  # Toggle drawing the rectangle on or off
    flip_horizontal = True  # Toggle horizontal flipping on or off

    # Create directory if it doesn't exist
    output_dir = 'camera_stream'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize frame counter
    frame_counter = 0

    for frame in show_camera_and_return_frames(camera_index, target_fps, effective_height, effective_width, apply_blur, show_rectangle, flip_horizontal):
        # Construct the filename
        filename = os.path.join(output_dir, f"{frame_counter}.png")
        
        # Save the frame as an image
        cv2.imwrite(filename, frame)
        
        # Print the shape of the frame for demonstration
        print("Returned frame shape:", frame.shape)
        
        # Increment the frame counter
        frame_counter += 1




