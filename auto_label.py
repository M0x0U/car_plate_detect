from ultralytics import YOLO
import cv2
import os
import shutil
import random
from pathlib import Path

# ==================== 路径配置 ====================
SOURCE_FOLDER = "car_plates/images"          # 你的图片文件夹
DATASET_ROOT = "dataset"         # 输出数据集

# 目录结构（完全匹配老师要求）
TRAIN_IMG = "train/images"
VAL_IMG   = "val/images"
TEST_IMG  = "test/images"
TRAIN_LABEL = "train/labels"
VAL_LABEL   = "val/labels"
TEST_LABEL  = "test/labels"

# ==================================================

# 创建目录
for folder in [
    os.path.join(DATASET_ROOT, TRAIN_IMG),
    os.path.join(DATASET_ROOT, VAL_IMG),
    os.path.join(DATASET_ROOT, TEST_IMG),
    os.path.join(DATASET_ROOT, TRAIN_LABEL),
    os.path.join(DATASET_ROOT, VAL_LABEL),
    os.path.join(DATASET_ROOT, TEST_LABEL),
]:
    os.makedirs(folder, exist_ok=True)

# 加载开源车牌检测器
model = YOLO("license_plate_detector.pt")

# 获取所有图片
img_paths = list(Path(SOURCE_FOLDER).glob("*.jpg")) + list(Path(SOURCE_FOLDER).glob("*.png"))
random.shuffle(img_paths)

# 划分 训练/验证/测试
total = len(img_paths)
train_num = int(total * 0.7)
val_num = int(total * 0.2)
test_num = total - train_num - val_num

train_imgs = img_paths[:train_num]
val_imgs   = img_paths[train_num : train_num+val_num]
test_imgs  = img_paths[train_num+val_num : ]

print(f"总图片：{total}")
print(f"训练集：{train_num}")
print(f"验证集：{val_num}")
print(f"测试集：{test_num}")

# 处理函数
def process(imgs, img_dir, label_dir):
    for img_path in imgs:
        print(f'Processing {img_path}')
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        name = img_path.stem

        # 推理车牌
        results = model(img, verbose=False)

        # 生成 YOLO 标签
        label_path = os.path.join(DATASET_ROOT, label_dir, f"{name}.txt")
        with open(label_path, "w") as f:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    # 转 YOLO 格式
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # 只标注车牌 → 类别 0
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # 复制图片
        shutil.copy(img_path, os.path.join(DATASET_ROOT, img_dir, img_path.name))

# 开始处理
process(train_imgs, TRAIN_IMG, TRAIN_LABEL)
process(val_imgs,   VAL_IMG,   VAL_LABEL)
process(test_imgs,  TEST_IMG,  TEST_LABEL)

print("\n✅ 数据集已生成完成！")
print("目录结构：")
print("dataset/")
print("  ├─ train/images  +  train/labels")
print("  ├─ val/images    +  val/labels")
print("  └─ test/images   +  test/labels")