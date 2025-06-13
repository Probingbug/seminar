import cv2
import numpy as np

# Load the full Moon surface image (big image)
moon_surface = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)
if moon_surface is None:
    print("Error loading moon surface image.")
    exit()

# Load the template (camera view or smaller screenshot)
template = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Screenshots for testing/equi.png', cv2.IMREAD_GRAYSCALE)
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

# Variables to track the best match
best_match_value = 0
best_match_coords = (0, 0)

while y + window_h <= moon_surface.shape[0] and x + window_w <= moon_surface.shape[1]:
    # Crop the current frame (simulate camera frame)
    frame = moon_surface[y:y+window_h, x:x+window_w]

    # Match the template in the frame
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # Draw rectangle (for visualization if needed)
    top_left = max_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame_color, top_left, bottom_right, (0,255,0), 2)

    # Display max value on the frame
    cv2.putText(frame_color, f'Max Value: {max_val:.2f}', (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame (show for a brief moment to simulate real-time)
    cv2.imshow('Camera View', frame_color)

    # Update the best match if this is the highest value so far
    if max_val > best_match_value:
        best_match_value = max_val
        best_match_coords = (x, y)

    # Update window position (simulate rightward camera movement)
    x += step_size
    if x + window_w > moon_surface.shape[1]:
        x = 0
        y += step_size

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After traversal, show the best match result on the full moon surface image
best_x, best_y = best_match_coords
frame_with_best_match = moon_surface.copy()

# Draw a rectangle around the best match on the original image
cv2.rectangle(frame_with_best_match, 
              (best_x, best_y), 
              (best_x + window_w, best_y + window_h), 
              (0, 255, 0), 2)

# Display best match coordinates and value on the image
cv2.putText(frame_with_best_match, 
            f'Best Match: {best_match_value:.2f}', 
            (best_x, best_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 255, 0), 2)

# Show the final image with the match result
cv2.imshow('Final Result', frame_with_best_match)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Best match found at coordinates: {best_match_coords} with match value: {best_match_value}")