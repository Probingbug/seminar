import cv2
import numpy as np

# Load images
large_img = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', 0)
template = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Screenshots for testing/equi.png', 0)

# Apply Canny Edge Detection
edges_large = cv2.Canny(large_img, 50, 150)
# cv2.imshow('canny',edges_large)
edges_template = cv2.Canny(template, 50, 150)

# Find contours
contours_large, _ = cv2.findContours(edges_large, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('contour_large',contours_large)
contours_template, _ = cv2.findContours(edges_template, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Ensure template has at least one contour
if len(contours_template) == 0:
    print("No contours found in template image")
    exit()




# Use matchShapes to find the best contour match
best_match = None
best_score = float('inf')
best_rect = None

# Loop through all template contours
for temp_cnt in contours_template:
    for cnt in contours_large:
        score = cv2.matchShapes(temp_cnt, cnt, 1, 0.0)  # Hu Moments
        print(f"Match Score: {score}")  # Debugging
        if score < best_score:
            best_score = score
            best_match = cnt
            best_rect = cv2.boundingRect(cnt)  # Bounding Box

# Draw the best matching contour and bounding box
if best_match is not None and best_rect is not None:
    x, y, w, h = best_rect
    print(f"Bounding Box Coordinates: x={x}, y={y}, w={w}, h={h}")  # Debugging

    # Green Rectangle Draw
    cv2.rectangle(large_img, (x, y), (x + w, y + h), (255, 0, 0), 5)  # Thickness 5 pixels

    # Show Image
    cv2.imshow("Detected Region", large_img)
    
    cv2.waitKey(0)
    
else:
    print("No matching region found")