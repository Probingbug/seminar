import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from sklearn.metrics import mean_squared_error
from math import sqrt

# Step 1: Generate a dummy grayscale image (you can replace this with your Moon image)
img_main =  cv2.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_3.png', cv2.IMREAD_GRAYSCALE)

h, w = img_main.shape

# Step 2: Generate random trajectory points
np.random.seed(None)
num_points = 12
given_traj = [(np.random.randint(60, w - 60), np.random.randint(60, h - 60)) for _ in range(num_points)]

# Step 3: Simulate shifted frames (as if the camera is moving)
frames = []
for i in range(num_points):
    dx = np.random.randint(-5, 5)
    dy = np.random.randint(-5, 5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_img = cv2.warpAffine(img_main, M, (w, h))
    frames.append(shifted_img)

# Step 4: Crop templates at trajectory positions
template_size = 40
templates = []
for (x, y) in given_traj:
    template = img_main[y - template_size//2:y + template_size//2,
                        x - template_size//2:x + template_size//2]
    templates.append(template)

# Step 5: Match templates using Template Matching
estimated_traj = []
start_time = time.time()
for i in range(num_points):
    template = templates[i]
    frame = frames[i]
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    center_x = max_loc[0] + template.shape[1] // 2
    center_y = max_loc[1] + template.shape[0] // 2
    estimated_traj.append((center_x, center_y))
end_time = time.time()

rmse = sqrt(mean_squared_error(given_traj, estimated_traj))

# Step 6: Animate the trajectory tracking
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2RGB))
line_given, = ax.plot([], [], color='red', linewidth=2, label='Given Trajectory')
line_est, = ax.plot([], [], color='blue', linewidth=2, label='Estimated Trajectory')
ax.set_title("Trajectory Tracking Animation")
ax.axis('off')
ax.legend()

given_x, given_y = zip(*given_traj)
est_x, est_y = [], []

def init():
    line_given.set_data([], [])
    line_est.set_data([], [])
    im.set_data(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2RGB))
    return im, line_given, line_est

def update(frame_idx):
    est_x.append(estimated_traj[frame_idx][0])
    est_y.append(estimated_traj[frame_idx][1])
    frame_img = cv2.cvtColor(frames[frame_idx], cv2.COLOR_GRAY2RGB)

    for i in range(1, frame_idx + 1):
        cv2.line(frame_img, given_traj[i - 1], given_traj[i], (0, 255, 0), 2)
        cv2.line(frame_img, estimated_traj[i - 1], estimated_traj[i], (255, 0, 0), 2)

    im.set_data(frame_img)
    line_given.set_data(given_x[:frame_idx + 1], given_y[:frame_idx + 1])
    line_est.set_data(est_x, est_y)
    return im, line_given, line_est

ani = animation.FuncAnimation(fig, update, frames=num_points, init_func=init,
                              interval=800, blit=True, repeat=False)

plt.suptitle(f"RMSE: {rmse:.2f} px | Time: {end_time - start_time:.2f} sec", fontsize=10)
plt.tight_layout()
plt.show()