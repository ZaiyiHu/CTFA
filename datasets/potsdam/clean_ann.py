import os
import cv2
import numpy as np

# 关系映射表
color_table = {
    (0, 1): 0,
    (2,): 1,
    (3,): 2,
    (4,): 3,
    (5,): 4,
    (255,): 5,
}


def replace_pixel_values(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 将颜色关系映射到新的值
    for colors, new_value in color_table.items():
        mask = np.isin(image, colors)
        image[mask] = new_value

    return image


def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            replaced_image = replace_pixel_values(input_path)
            cv2.imwrite(output_path, replaced_image)


if __name__ == "__main__":
    input_folder = '/data1/zaiyihu/Datasets/vaihingen_IRRG_wiB_512_512_dl/ann_dir/test'  # 输入文件夹
    output_folder = '/data1/zaiyihu/Datasets/vaihingen_IRRG_wiB_512_512_dl/ann_dir/test_new'  # 输出文件夹

    main(input_folder, output_folder)
    print('Done!')