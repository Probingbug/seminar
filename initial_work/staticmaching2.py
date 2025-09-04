import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale images
img_main = cv2.imread('Moon_image_dataset/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)
img_template = cv2.imread('Moon_image_dataset/Screenshots for testing/Screenshot 2025-09-02 at 6.56.31 PM.png', cv2.IMREAD_GRAYSCALE)

if img_main is None or img_template is None:
    raise ValueError("Check your image paths!")

# ORB (Oriented FAST and Rotated BRIEF) feature detector

orb = cv2.ORB_create(nfeatures=100000)

# Keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img_template, None)
kp2, des2 = orb.detectAndCompute(img_main, None)

# Use BFMatcher with Hamming distance (best for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# knnMatch returns k best matches
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches for visualization
result_img = cv2.drawMatches(img_template, kp1, img_main, kp2, good_matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print(len(good_matches))

plt.figure(figsize=(16, 8))
plt.imshow(result_img)
plt.title(f"Good Matches: {len(good_matches)}")
plt.axis('off')
plt.show()

# Optional: detect the matching region using homography
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img_template.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw the polygon on the main image
    img_main_color = cv2.cvtColor(img_main, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_main_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Calculate the position of the top-left corner of the detected region
    top_left = np.int32(dst[0][0])
    bottom_right = np.int32(dst[2][0])

    # Calculate the center position of the detected region
    detected_center = (top_left + bottom_right) // 2

    # Let's assume you know the actual position of the template (in pixels)
    # For example, (actual_x, actual_y) is the true known position of the top-left corner of the template in the main image
    # You can update this with the real position of the template in the large image if known
    actual_x, actual_y = 100, 150  # Replace with your known actual position

    # Calculate error in position (Euclidean distance between actual and detected positions)
    error_x = detected_center[0] - actual_x
    error_y = detected_center[1] - actual_y
    error = np.sqrt(error_x**2 + error_y**2)

    print(f"Detected Center Position: {detected_center}")
    print(f"Position Error: {error:.2f} pixels")

    # Annotate the coordinates and error on the image
    cv2.putText(img_main_color, f"Detected Position: {detected_center}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_main_color, f"Error: {error:.2f} px", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the final image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_main_color)
    plt.title("Detected Crater using ORB + BFMatcher")
    plt.axis('off')
    plt.show()

else:
    print("Not enough good matches to compute homography.")
