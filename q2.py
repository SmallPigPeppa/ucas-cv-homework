import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
cover_image = cv2.imread('cv_cover.jpg')  # 需要放置的图片
desk_image = cv2.imread('cv_desk.png')   # 背景图片
hp_cover = cv2.imread('hp_cover.jpg')    # 新加入的图片

# 调整 hp_cover 尺寸为 cover_image 的大小
hp_cover_resized = cv2.resize(hp_cover, (cover_image.shape[1], cover_image.shape[0]))

# 转为灰度图像
gray_cover = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
gray_desk = cv2.cvtColor(desk_image, cv2.COLOR_BGR2GRAY)

# 使用 ORB 特征检测器来找到关键点和描述符
orb = cv2.ORB_create()

# 找到关键点和描述符
keypoints_cover, descriptors_cover = orb.detectAndCompute(gray_cover, None)
keypoints_desk, descriptors_desk = orb.detectAndCompute(gray_desk, None)

# 使用暴力匹配器 (Brute-Force Matcher) 匹配描述符
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_cover, descriptors_desk)

# 按照距离对匹配结果进行排序
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点的位置
points_cover = np.zeros((len(matches), 2), dtype=np.float32)
points_desk = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points_cover[i] = keypoints_cover[match.queryIdx].pt
    points_desk[i] = keypoints_desk[match.trainIdx].pt

# 使用 RANSAC 估算单应性矩阵
H, mask = cv2.findHomography(points_cover, points_desk, cv2.RANSAC, 5.0)

# 获取 cv_cover.jpg 的四个角点
height, width = cover_image.shape[:2]
corners_cover = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

# 使用单应性矩阵计算转换后的角点
corners_desk = cv2.perspectiveTransform(corners_cover[None, :, :], H)[0]

# 绘制匹配的特征点
img_matches = cv2.drawMatches(cover_image, keypoints_cover, desk_image, keypoints_desk, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 绘制背景图像中的绿色框，标出匹配区域
desk_with_box = desk_image.copy()
cv2.polylines(desk_with_box, [np.int32(corners_desk)], isClosed=True, color=(0, 255, 0), thickness=2)

# 使用单应性矩阵将 cv_cover.jpg 映射到 cv_desk.png 上
height, width, channels = desk_image.shape
warped_cover = cv2.warpPerspective(hp_cover_resized, H, (width, height))  # 使用调整过大小的 hp_cover

# 将 cv_cover.jpg 叠加到 cv_desk.png 上
overlay = desk_image.copy()
mask = warped_cover != 0  # 创建掩码，排除完全黑色的区域（背景区域）

# 在匹配的位置将两张图像融合
overlay[mask] = warped_cover[mask]

# 保存最终结果
cv2.imwrite('q2_final_result.png', overlay)
