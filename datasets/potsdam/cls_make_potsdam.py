import os
import cv2
import numpy as np
from tqdm import tqdm

# 设置类别数
num_classes = 6  # 增加一个背景类别


# color_table = {
#     (255, 255, 255): 0,       # background
#     (0, 0, 255): 1,     # imp surf
#     (0, 255, 255): 2,        # car
#     (0, 255, 0): 3,    # tree
#     (255, 255, 0): 4,      # low vegetation
#     (255, 0,  0): 5,     # building
# }
color_table = {
    (1, 1, 1): 0,       # imp surf white  [255,255,255]
    (2, 2, 2): 1,     # building deep blue [255,0,0]
    (3, 3, 3): 2,        # low vegetation light blue [255,255,0]
    (4, 4, 4): 3,    # tree green [0,255,0]
    (5, 5, 5): 4,      # car yellow [0,255,255]
    (255, 255,  255): 5,     # imp surf red [0,0,255]
}
# 获取所有标签图像文件名
data_dir = '/data1/zaiyihu/Datasets/potsdam_IRRG_wiB_512_256_dl/ann_dir'
train_label_dir = os.path.join(data_dir, 'train')
test_label_dir = os.path.join(data_dir, 'test')
train_label_files = os.listdir(train_label_dir)
test_label_files = os.listdir(test_label_dir)

# 遍历所有标签图像，生成对应的One-hot分类
class_dict = {}
for label_files, label_dir in [(train_label_files, train_label_dir), (test_label_files, test_label_dir)]:
    for label_file in tqdm(label_files, desc='Processing {}'.format(label_dir)):
        label_path = os.path.join(label_dir, label_file)
        if label_path.startswith('4_12'):
            continue
        label_img = cv2.imread(label_path)
        label_class = np.zeros(num_classes, dtype=np.float32)
        for color, cls_idx in color_table.items():
            mask = np.all(label_img == color, axis=-1)
            label_class[cls_idx] = np.any(mask)
        class_dict[label_file.split('.')[0]] = label_class

# 保存为npy格式
output_path = os.path.join(data_dir, 'cls_labels_onehot_potsdam.npy')
np.save(output_path, class_dict)

print('Done.')
# import cv2
# import numpy as np
# def main():
#     # 读取图片
#     image_path = '/data1/zaiyihu/Datasets/potsdam_IRRG_wiB_512_256_dl/ann_dir/train/4_12_103_2048_4608_2560_5120.png'  # 替换为你的图片路径
#     image = cv2.imread(image_path)
#
#     if image is None:
#         print("无法读取图片")
#         return
#
#     # 显示图片
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
