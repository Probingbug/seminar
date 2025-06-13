import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale images
img_main = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)
img_template = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Screenshots for testing/equi.png', cv2.IMREAD_GRAYSCALE)

if img_main is None or img_template is None:
    raise ValueError("Check your image paths!")

# ORB ( oriented Fast and Rotated Brief) feature detector
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

plt.figure(figsize=(16, 8))
plt.imshow(result_img)
plt.title(f"Good Matches: {len(good_matches)}")
plt.axis('off')
plt.show()

# Optional: detect the matching region using homography
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img_template.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img_main_color = cv2.cvtColor(img_main, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_main_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_main_color)
    plt.title("Detected Crater using ORB + BFMatcher")
    plt.axis('off')
    plt.show()
else:
    print("Not enough good matches to compute homography.")