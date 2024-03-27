# import os
# import cv2
# import numpy as np
#
# # 强度映射表
# intensity_table = {
#     0: 1,
#     1: 2,
#     2: 3,
#     3: 4,
#     4: 5,
#     5: 6,
#     6: 7,
# }
#
#
# def replace_pixel_values(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     new_image = np.copy(image)
#
#     for intensity in intensity_table:
#         mask = (image == intensity)
#         new_value = intensity_table[intensity]
#         new_image[mask] = new_value
#
#     return new_image
#
#
# def main(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#
#             replaced_image = replace_pixel_values(input_path)
#             cv2.imwrite(output_path, replaced_image)
#
#
# if __name__ == "__main__":
#     input_folder = '/data1/zaiyihu/Datasets/deepglobe_512_512/ann_dir/train'
#     output_folder = '/data1/zaiyihu/Datasets/deepglobe_512_512/ann_dir/train_new'
#
#     main(input_folder, output_folder)
#     print('Done!')
# 读取第一个文件
# 读取第一个文件
import os
# 读取第一个文件
file1_path = "/home/zaiyihu/CodeSpace/CTFA-main/datasets/deepglobe/train_ori.txt"
with open(file1_path, 'r') as file1:
    lines1 = file1.readlines()

# 读取第二个文件
file2_path = "/home/zaiyihu/CodeSpace/CTFA-main/datasets/deepglobe/train_noblack.txt"
with open(file2_path, 'r') as file2:
    prefixes_to_remove = set(os.path.splitext(os.path.basename(line.strip()))[0] for line in file2)

# 保留第一个文件中不包含第二个文件中文件名前缀的行
filtered_lines = [line for line in lines1 if os.path.splitext(os.path.basename(line.strip()))[0] not in prefixes_to_remove]

# 写入新文件
output_file_path = "/home/zaiyihu/CodeSpace/CTFA-main/datasets/deepglobe/train_filtered.txt"
with open(output_file_path, 'w') as output_file:
    output_file.writelines(filtered_lines)

print("Filtered lines written to: {}".format(output_file_path))

