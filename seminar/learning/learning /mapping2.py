import cv2
import numpy as np

# Load the larger Moon surface image and the crater template
moon_image = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)
crater_template = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Screenshots for testing/equi.png', cv2.IMREAD_GRAYSCALE)

# Template matching
result = cv2.matchTemplate(moon_image, crater_template, cv2.TM_CCOEFF_NORMED)

# Threshold to filter strong matches
threshold = 0.8
locations = np.where(result >= threshold)

# Draw rectangles on the detected crater positions
h, w = crater_template.shape
output_image = cv2.cvtColor(moon_image, cv2.COLOR_GRAY2BGR)  # Convert to color for drawing
for pt in zip(*locations[::-1]):
    cv2.rectangle(output_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# Display the result using OpenCV
cv2.imshow("Detected Craters", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()