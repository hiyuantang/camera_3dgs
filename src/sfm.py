import cv2
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import keyboard

# Helper function to initialize the 3D plot
def initialize_3d_plot():
    fig = go.Figure(data=[go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='markers',
        marker=dict(size=2)
    )])
    fig.show()
    return fig

# Helper function to update 3D points in the existing Plotly figure
def update_3d_points(fig, points):
    fig.data[0].x = points[:, 0]
    fig.data[0].y = points[:, 1]
    fig.data[0].z = points[:, 2]
    fig.show()

# Function to process images and compute 3D points
def process_images(image_paths):
    # Initialize variables
    points_3d = []
    
    # Initialize the 3D plot
    fig = initialize_3d_plot()
    
    # Loop through the image sets (0,1,2), (1,2,3), ...
    for i in range(len(image_paths) - 2):
        img1 = cv2.imread(image_paths[i])
        img2 = cv2.imread(image_paths[i + 1])
        img3 = cv2.imread(image_paths[i + 2])
        
        # Add your SfM computation logic here
        # For simplicity, let's assume we have a function `compute_sfm`
        new_points = compute_sfm(img1, img2, img3)
        points_3d.append(new_points)
        
        # Update the display
        all_points = np.concatenate(points_3d)
        update_3d_points(fig, all_points)
        
        # Check if 'q' key is pressed to quit
        if keyboard.is_pressed('q'):
            break

# Dummy SfM function (replace with actual SfM logic)
def compute_sfm(img1, img2, img3):
    # Dummy 3D points
    return np.random.rand(10, 3) * 10

# Example usage
image_paths = ["0.png", "1.png", "2.png", "3.png", "4.png"]
process_images(image_paths)


