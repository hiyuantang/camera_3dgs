import cv2
import numpy as np
import plotly.graph_objects as go

# Given metadata
focal_length_mm = 26
sensor_width_mm = 5.76
image_width_px = 4032
image_height_px = 3024

# Calculate focal length in pixels
fx = (focal_length_mm * image_width_px) / sensor_width_mm
fy = fx  # Assuming square pixels
cx = image_width_px / 2
cy = image_height_px / 2

# Construct intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

print("Intrinsic Matrix K:\n", K)

# Load images
img1 = cv2.imread('camera_stream/0.png', 0)
img2 = cv2.imread('camera_stream/1.png', 0)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect key points and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors using FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Store all good matches as per Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Compute the essential matrix
E = K.T @ F @ K

# Recover pose
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Triangulate points
pts1_hom = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
pts2_hom = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))
P1 = K @ P1
P2 = K @ P2

points_4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3D = points_4D_hom[:3] / points_4D_hom[3]


# Prepare data for Plotly
x = points_3D[0]
y = points_3D[1]
z = points_3D[2]

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=z,                # Set color to the Z values
        colorscale='Viridis',   # Choose a colorscale
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='3D Point Cloud from SfM',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show plot
fig.show()
