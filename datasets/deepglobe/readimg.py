import cv2
from matplotlib import pyplot as plt

def read_and_visualize_image(file_path):
    try:
        # 读入图像
        img = cv2.imread(file_path)

        if img is None:
            raise Exception("无法读取图像文件")

        # 将图像从 BGR 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 可视化图像
        plt.imshow(img_rgb)
        plt.axis('off')  # 不显示坐标轴
        plt.show()

        return img

    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 指定图像文件路径
image_path = "/data1/zaiyihu/Datasets/deepglobe_512_512/ann_dir/test/882_004_1936_0_2448_512.png"

# 读入并可视化图像
image = read_and_visualize_image(image_path)

# 如果成功读入图像，可以进行其他处理
if image is not None:
    # 在这里可以添加其他图像处理代码，如保存、转换等
    pass


