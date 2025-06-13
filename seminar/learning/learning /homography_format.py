# Optional: detect the matching region using homography
# if len(good_matches) > 10:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     h, w = img_template.shape
#     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)

#     img_main_color = cv2.cvtColor(img_main, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(img_main_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

#     plt.figure(figsize=(10, 6))
#     plt.imshow(img_main_color)
#     plt.title("Detected Crater using ORB + BFMatcher")
#     plt.axis('off')
#     plt.show()
# else:
#     print("Not enough good matches to compute homography.")