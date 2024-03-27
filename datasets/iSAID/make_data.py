import os
import random
import shutil

# 设置随机数种子，保证每次运行的结果都相同
random.seed(42)

# 定义图像文件夹路径和输出文件夹路径
data_dir = '/data1/zaiyihu/Datasets/iSAID_patches_512'
train_dir = os.path.join(data_dir, 'train', 'images')
test_dir = os.path.join(data_dir, 'val', 'images')
output_dir = '/data1/zaiyihu/Datasets/iSAID_patches_512/sampled_process'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def copy_pairs(pairs, src_dir, dst_dir):
    for pair in pairs:
        src_image_path = os.path.join(src_dir, pair)
        src_label_path = os.path.join(src_dir, pair.replace('.png', '_instance_color_RGB.png'))

        dst_image_path = os.path.join(output_dir, dst_dir, 'images', pair)
        dst_label_path = os.path.join(output_dir, dst_dir, 'labels', pair.replace('.png', '_instance_color_RGB.png'))
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
        shutil.copyfile(src_image_path, dst_image_path)
        shutil.copyfile(src_label_path, dst_label_path)


# 获取所有图像文件名，只保留图片文件名，而不包括对应的标签文件名
train_image_files = [f for f in os.listdir(train_dir) if '_instance_color_RGB.png' not in f]
test_image_files = [f for f in os.listdir(test_dir) if '_instance_color_RGB.png' not in f]

# 从训练集中随机抽取7500对作为训练集数据集，1653对作为验证集
train_val_pairs = random.sample(train_image_files, 9153)

train_pairs = train_val_pairs[:7500]
val_pairs = train_val_pairs[7500:]

# 从测试集中随机抽取1315对作为测试集
test_pairs = random.sample(test_image_files, 1315)

# 复制文件到输出文件夹中
copy_pairs(train_pairs, train_dir, 'train')
copy_pairs(val_pairs, train_dir, 'val')
copy_pairs(test_pairs, test_dir, 'test.txt')
