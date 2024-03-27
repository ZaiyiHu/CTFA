import os

data_dir = '/data1/zaiyihu/Datasets/iSAID_patches_512/sampled_process'

# 获取所有图像文件名
train_files = os.listdir(os.path.join(data_dir, 'train', 'images'))
val_files = os.listdir(os.path.join(data_dir, 'val', 'images'))
test_files = os.listdir(os.path.join(data_dir, 'test.txt', 'images'))

# 从文件名中提取P+数字部分
def extract_P_num(file_name):
    return file_name.split('.')[0]

# 写入train.txt
with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
    for file_name in train_files:
        P_num = extract_P_num(file_name)
        f.write(f'{P_num}\n')

# 写入val.txt
with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
    for file_name in val_files:
        P_num = extract_P_num(file_name)
        f.write(f'{P_num}\n')

# 写入test.txt
with open(os.path.join(data_dir, 'test.txt.txt'), 'w') as f:
    for file_name in test_files:
        P_num = extract_P_num(file_name)
        f.write(f'{P_num}\n')
