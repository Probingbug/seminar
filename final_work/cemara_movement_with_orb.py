# with orb brute force matcher


import cv2
import numpy as np

# Load the full Moon surface image (big image)
moon_surface = cv2.imread('Moon_image_dataset/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)
if moon_surface is None:
    print("Error loading moon surface image.")
    exit()

# Load the template (camera view or smaller screenshot)
template = cv2.imread('/Users/anupam/mtech /MTP work/Moon_image_dataset/Screenshots for testing/Screenshot 2025-09-04 at 8.42.04 PM.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error loading template image.")
    exit()

template_h, template_w = template.shape

# Define initial window position
x, y = 0, 0
window_w, window_h = template_w, template_h  # Window size matches template size

# Speed optimization
step_size = 20  # Increase step size to speed up
fps = 30  # Increased FPS
frame_time = 1 / fps

# ORB detector and BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Compute keypoints and descriptors for template once
kp_template, des_template = orb.detectAndCompute(template, None)

# Variables to track the best match
best_match_value = 0
best_match_coords = (0, 0)
best_accuracy = 0

while y + window_h <= moon_surface.shape[0] and x + window_w <= moon_surface.shape[1]:
    # Crop the current frame (simulate camera frame)
    frame = moon_surface[y:y+window_h, x:x+window_w]

    # orb = cv2.ORB_create(nfeatures=10000000)

    # ORB keypoints & descriptors for current frame
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    match_score = 0
    if des_frame is not None and len(des_frame) > 0:
        # Match descriptors
        matches = bf.match(des_template, des_frame)

        # Score = number of matches (you can also use sum of distances)
        match_score = len(matches)
        if len(kp_template)>0:
            accuracy = (match_score / len(kp_template)) * 100

    # Draw rectangle (for visualization if needed)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.putText(frame_color,f"Accuracy : {accuracy : .2f}%", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Camera View', frame_color)

    # Update the best match if this is the highest score so far
    if match_score > best_match_value:
        best_match_value = match_score
        best_match_coords = (x, y)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        

    # Update window position (simulate rightward camera movement)
    x += step_size
    if x + window_w > moon_surface.shape[1]:
        x = 0
        y += step_size

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After traversal, show the best match result on the full moon surface image
best_x, best_y = best_match_coords
frame_with_best_match = cv2.cvtColor(moon_surface.copy(), cv2.COLOR_GRAY2BGR)

# Draw a rectangle around the best match on the original image
cv2.rectangle(frame_with_best_match, 
              (best_x, best_y), 
              (best_x + window_w, best_y + window_h), 
              (0, 255, 0), 2)

# Display best match score on the image
cv2.putText(frame_with_best_match, 
            f'Best Accuracy: {best_accuracy:.2f}%', 
            # f"best matches : {best_match_value:.2f}",
            (best_x, best_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 255, 0), 2)

# Show the final image with the match result
cv2.imshow('Final Result', frame_with_best_match)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Best match found at coordinates: {best_match_coords} with match score: {best_match_value}")