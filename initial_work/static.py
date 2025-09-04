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

plt.figure(figsize=(16, 8))
plt.imshow(result_img)
plt.title(f"Good Matches: {len(good_matches)}")
plt.axis('off')
plt.show()

# Detect the matching region using homography
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img_template.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Convert main image to color
    img_main_color = cv2.cvtColor(img_main, cv2.COLOR_GRAY2BGR)

    # Draw predicted polygon (green)
    cv2.polylines(img_main_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Calculate predicted center
    pred_pts = dst.reshape(-1, 2)
    pred_center = tuple(np.mean(pred_pts, axis=0).astype(int))

    # Ground-truth (manually known)
    actual_x, actual_y = 100, 150  # Replace with your known actual position
    gt_tl = (actual_x, actual_y)
    gt_br = (actual_x + w, actual_y + h)
    gt_center = ((gt_tl[0] + gt_br[0]) // 2, (gt_tl[1] + gt_br[1]) // 2)

    # Draw ground truth rectangle (red)
    cv2.rectangle(img_main_color, gt_tl, gt_br, (0, 0, 255), 3)

    # Draw centers
    cv2.circle(img_main_color, gt_center, 5, (0, 0, 255), -1)
    cv2.circle(img_main_color, pred_center, 5, (0, 255, 0), -1)

    # Error calculation
    error = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
    diag = np.hypot(w, h)
    accuracy = max(0, 1 - error / diag) * 100

    print(f"Ground Truth Center: {gt_center}")
    print(f"Predicted Center:    {pred_center}")
    print(f"Error: {error:.2f} px")
    print(f"Accuracy: {accuracy:.2f}%")

    # Annotate on image
    cv2.putText(img_main_color, f"Error: {error:.2f}px", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_main_color, f"Accuracy: {accuracy:.2f}%", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_main_color, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth (Red) vs Predicted (Green)")
    plt.axis('off')
    plt.show()

else:
    print("Not enough good matches to compute homography.")