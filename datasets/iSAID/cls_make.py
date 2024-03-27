import os
import cv2
import numpy as np
from tqdm import tqdm

# 设置类别数
num_classes = 16  # 增加一个背景类别

# 定义颜色表。 是真的一个一个看出来的。
color_table = {
    (0, 127, 255): 14,    # plane
    (0, 0, 127): 9,       # small_vehicle
    (0, 127, 127): 8,     # large_vehicle
    (0, 0, 63): 1,        # ship
    (0, 100, 155): 15,    # harbor
    (0, 63, 127): 4,      # tennis_court
    (0, 0, 255): 11,      # swimming_pool
    (0, 63, 255): 3,      # baseball_diamond
    (0, 63, 63): 2,       # stroage_tank
    (0, 127, 63): 7,      # bridge
    (0, 63, 0): 6,        # Ground_Track_Field
    (0, 0, 191): 10,      # helicopter
    (0, 63, 191): 5,      # basketball_court
    (0, 127, 191): 13,    # Soccer_ball_field
    (0, 191, 127): 12     # Roundabout
}

# 获取所有标签图像文件名
data_dir = '/data1/zaiyihu/Datasets/iSAID_patches_512/sampled_process'
train_label_dir = os.path.join(data_dir, 'train', 'labels')
val_label_dir = os.path.join(data_dir, 'val', 'labels')
test_label_dir = os.path.join(data_dir, 'test', 'labels')
train_label_files = os.listdir(train_label_dir)
val_label_files = os.listdir(val_label_dir)
test_label_files = os.listdir(test_label_dir)

# 遍历所有标签图像，生成对应的One-hot分类
class_dict = {}
for label_files, label_dir in [(train_label_files, train_label_dir), (val_label_files, val_label_dir), (test_label_files, test_label_dir)]:
    for label_file in tqdm(label_files, desc='Processing {}'.format(label_dir)):
        label_path = os.path.join(label_dir, label_file)
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label_class = np.zeros(num_classes, dtype=np.float32)
        for color, cls_idx in color_table.items():
            mask = np.all(label_img == color, axis=-1)
            label_class[cls_idx] = np.any(mask)
        class_dict[label_file.split('.')[0]] = label_class

# 保存为npy格式
output_path = os.path.join(data_dir, 'cls_labels_onehot_iSAID_2.npy')
np.save(output_path, class_dict)

print('Done.')
