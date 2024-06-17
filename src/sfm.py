import cv2
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

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
        
        # Compute SfM
        new_points = compute_sfm(img1, img2, img3)
        points_3d.append(new_points)
        
        # Update the display
        all_points = np.concatenate(points_3d)
        update_3d_points(fig, all_points)

# Function to compute 3D points using SfM with OpenCV
def compute_sfm(img1, img2, img3):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB key points and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    kp3, des3 = orb.detectAndCompute(gray3, None)
    
    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches12 = bf.match(des1, des2)
    matches23 = bf.match(des2, des3)
    
    # Extract matched key points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches12]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches12]).reshape(-1, 1, 2)
    pts3 = np.float32([kp2[m.queryIdx].pt for m in matches23]).reshape(-1, 1, 2)
    pts4 = np.float32([kp3[m.trainIdx].pt for m in matches23]).reshape(-1, 1, 2)
    
    # Estimate the essential matrix between image pairs
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover pose from the essential matrix
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2)
    
    # Triangulate points to get 3D coordinates
    points_3d_homogeneous = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1, pts2)
    points_3d = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T)
    
    return points_3d.reshape(-1, 3)

# Example usage
image_paths = ["camera_stream/0.png", "camera_stream/1.png", "camera_stream/2.png", 
               "camera_stream/3.png", "camera_stream/4.png", "camera_stream/5.png", 
               "camera_stream/6.png", "camera_stream/7.png", "camera_stream/8.png", 
               "camera_stream/9.png"]
process_images(image_paths)



