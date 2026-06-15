import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# 设置标签文件路径
# label_dir = r'/data1/cjp/Datasets/MFNet/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/PST900/labels_train'  # L
label_dir = r'/data1/cjp/Datasets/NYU_Depth_V2/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/SUN-RGBD/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/Cityscapes/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/EventScape/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/KITTI-360/labels_train'  # L
# label_dir = r'/data1/cjp/Datasets/DELIVER/labels_train'  # RGBA
# label_dir = r'/data1/cjp/Datasets/ZJU-RGB-P/labels_train'  # P

# 存储所有标签的唯一值
all_unique_labels = set()
label_frequencies = Counter()

# 可视化一张标签图像
def visualize_label(label_array):
    """可视化标签图像，为不同类别分配不同颜色并添加数字标注"""
    # 创建一个彩色图像
    colored_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

    # 为每个类别分配不同的颜色
    for label_id in np.unique(label_array):
        if label_id == 255:  # 忽略标签
            color = [0, 0, 0]  # 黑色
        else:
            # 为每个类别生成一个独特的颜色
            np.random.seed(label_id)
            color = np.random.randint(0, 256, 3).tolist()

        # 将对应标签位置设置为该颜色
        colored_label[label_array == label_id] = color

    # 使用OpenCV添加数字标注
    colored_label_cv = colored_label.copy()
    # 为每个类别找到中心位置添加数字标签
    for label_id in np.unique(label_array):
        if label_id == 255:  # 忽略背景
            continue
        # 创建掩码并找到连通区域
        mask = (label_array == label_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        # 在每个连通区域添加标签号
        for i in range(0, num_labels):
            x, y = int(centroids[i][0]), int(centroids[i][1])
            # 使用白色文本和黑色边框确保在任何背景下都可见
            cv2.putText(colored_label_cv, str(label_id), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(colored_label_cv, str(label_id), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示彩色标签图像
    colored_image = Image.fromarray(colored_label_cv)
    # colored_image.show()
    return colored_image

# 遍历标签目录中的所有图像
for filename in os.listdir(label_dir):
    if filename.endswith('.png'):  # 根据实际文件格式调整
        label_path = os.path.join(label_dir, filename)

        # 加载标签图像
        label_image = Image.open(label_path)
        print(f"Loading {filename}...")
        print(f"Image mode: {label_image.mode}")
        label_array = np.array(label_image)
        print(f"Image shape: {label_array.shape}")

        # 检查唯一标签
        unique_labels = np.unique(label_array)
        print(f"Unique labels: {unique_labels}, label_nums: {len(unique_labels)}")
        all_unique_labels.update(unique_labels)

        # 统计每个标签的频率
        label_count = Counter(label_array.flatten())
        label_frequencies.update(label_count)

        # # 可视化标签图像
        # print(f"Visualizing the first label image: {filename}")
        # visualize_label(label_array)
        # 如果需要保存可视化结果
        # vis_path = os.path.join(os.path.dirname(label_dir), 'visualization')
        # os.makedirs(vis_path, exist_ok=True)
        # colored_image = visualize_label(label_array)
        # colored_image.save(os.path.join(vis_path, f'vis_{filename}'))
        # print(f"Saved visualization to {os.path.join(vis_path, f'vis_{filename}')}")

# 将集合转换为排序列表
sorted_unique_labels = sorted(list(all_unique_labels))
print(f"所有唯一标签值（按升序排列）: {sorted_unique_labels}")
print(f"唯一标签总数: {len(sorted_unique_labels)}")

