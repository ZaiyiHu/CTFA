import os
import cv2
import numpy as np
from tqdm import tqdm

# 设置类别数
num_classes = 7  # 增加一个背景类别

color_table = {
    (0,0,0): 0,       # urban blue
    (1,1,1): 1,     # agriculture yellow
    (2,2,2): 2,        # rangeland pink
    (3,3,3): 3,    # forest green
    (4,4,4): 4,      # water deep blue
    (5,5,5): 5,     # barren white
    (6,6,6): 6,      # unknown dark
}
# 获取所有标签图像文件名
data_dir = '/data1/zaiyihu/Datasets/deepglobe_512_512/ann_dir'
train_label_dir = os.path.join(data_dir, 'train')
test_label_dir = os.path.join(data_dir, 'test')
train_label_files = os.listdir(train_label_dir)
test_label_files = os.listdir(test_label_dir)

# 遍历所有标签图像，生成对应的One-hot分类
class_dict = {}
for label_files, label_dir in [(train_label_files, train_label_dir), (test_label_files, test_label_dir)]:
    for label_file in tqdm(label_files, desc='Processing {}'.format(label_dir)):
        label_path = os.path.join(label_dir, label_file)
        label_img = cv2.imread(label_path)
        label_class = np.zeros(num_classes, dtype=np.float32)
        for color, cls_idx in color_table.items():
            mask = np.all(label_img == color, axis=-1)
            label_class[cls_idx] = np.any(mask)
        class_dict[label_file.split('.')[0]] = label_class

# 保存为npy格式
output_path = os.path.join(data_dir, 'cls_labels_onehot_deepglobe_revised.npy')
np.save(output_path, class_dict)

print('Done.')
# import cv2
# import numpy as nps
# def main():
#     # 读取图片
#     # image_path = '/data1/zaiyihu/Datasets/vaihingen_IRRG_wiB_512_512_dl/ann_dir/train/area1_000_0_0_512_512.png'  # 替换为你的图片路径
#     # image_path = '/data1/zaiyihu/Datasets/vaihingen_IRRG_wiB_512_512_dl/ann_dir/train_new/area1_000_0_0_512_512.png'
#     # image_path = '/data1/zaiyihu/Datasets/vaihingen_IRRG_wiB_512_512_dl/ann_dir/test/area10_004_512_512_1024_1024.png' # 替换为你的图片路径
#     # image_path = '/home/zaiyihu/CodeSpace/atws-main/results/2023-8-19-17-03-first-afa-vaihingen/test/prediction_cmap/area2_000_0_0_512_512.png'
#     # image_path = '/home/zaiyihu/CodeSpace/atws-main/results/2023-8-19-17-03-first-afa-vaihingen/test/prediction/area1_004_512_512_1024_1024.png'
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
