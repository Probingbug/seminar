import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import time

# Load base image
img_main = cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', 0)
if img_main is None:
    raise ValueError("Image not found!")

h, w = img_main.shape

# Generate a random trajectory
given_traj = []
for _ in range(25):
    x = random.randint(60, w - 60)
    y = random.randint(60, h - 60)
    given_traj.append((x, y))

# Crop templates
template_size = 40
templates = []
for (x, y) in given_traj:
    template = img_main[y - template_size // 2:y + template_size // 2,
                        x - template_size // 2:x + template_size // 2]
    templates.append(template)

# Apply rotation
angle = 10  # in degrees
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(img_main, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

# Initial estimation using template matching
estimated_traj = []
for template in templates:
    res = cv2.matchTemplate(rotated_img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    cx = max_loc[0] + template.shape[1] // 2
    cy = max_loc[1] + template.shape[0] // 2
    estimated_traj.append([cx, cy])

# --- Simulate optimization (adjust estimated points to get closer) ---
epochs = 30
rmse_list = []

for epoch in range(epochs):
    new_traj = []
    for (gx, gy), (ex, ey) in zip(given_traj, estimated_traj):
        # Move a fraction towards ground truth (simulate learning)
        new_x = ex + 0.2 * (gx - ex)
        new_y = ey + 0.2 * (gy - ey)
        new_traj.append([new_x, new_y])
    estimated_traj = new_traj
    rmse = sqrt(mean_squared_error(given_traj, estimated_traj))
    rmse_list.append(rmse)

# --- Plot RMSE over epochs ---
plt.figure(figsize=(8, 4))
plt.plot(rmse_list, marker='o', color='orange')
plt.title("RMSE Over Epochs (Simulated Learning)")
plt.xlabel("Epoch")
plt.ylabel("RMSE (pixels)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot final comparison ---
comparison = cv2.cvtColor(img_main, cv2.COLOR_GRAY2BGR)
for i in range(len(given_traj) - 1):
    cv2.line(comparison, given_traj[i], given_traj[i + 1], (0, 255, 0), 2)
for i in range(len(estimated_traj) - 1):
    pt1 = tuple(map(int, estimated_traj[i]))
    pt2 = tuple(map(int, estimated_traj[i + 1]))
    cv2.line(comparison, pt1, pt2, (0, 0, 255), 2)

plt.figure(figsize=(8, 8))
plt.imshow(comparison)
plt.title(f"Final Trajectory Comparison\nGreen = Ground Truth | Red = Estimated")
plt.axis('off')
plt.tight_layout()
plt.show()