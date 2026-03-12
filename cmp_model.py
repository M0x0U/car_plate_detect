from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random  # ✅ 导入随机模块
import os      # ✅ 新增：导入os模块用于路径处理

# 1. 定义路径
MODEL_PATH1 = '/home/mx2123/mx2123/runspace_mx_yolo/run_plate/license_plate_detector.pt' # Baseline
MODEL_PATH2 = '/home/mx2123/mx2123/runspace_mx_yolo/run_plate/runs/detect/plate_detector3/weights/best.pt' # Ours
DATA_PATH = '/home/mx2123/mx2123/runspace_mx_yolo/run_plate/dataset'

# ✅ 新增：定义可视化结果保存的目录
VIS_SAVE_DIR = 'comparison_visualizations'
Path(VIS_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# 2. 加载模型
print("正在加载模型...")
model1 = YOLO(MODEL_PATH1)
model2 = YOLO(MODEL_PATH2)

# 3. 获取并随机抽取图片
print("正在读取数据集...")
img_dir = Path(DATA_PATH)
img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []
for ext in img_exts:
    image_paths.extend(list(img_dir.rglob(ext))) 

if len(image_paths) == 0:
    raise ValueError(f"在 {DATA_PATH} 目录下没有找到任何图片！请检查路径。")

# 将整个数据集的图片路径随机打乱（洗牌）
random.seed(42)  # 设置随机种子，确保每次运行抽到的100张图是一样的
random.shuffle(image_paths)

# 截取打乱后的前100张
limit = 100
images = image_paths[:limit]
print(f"总数据集包含 {len(image_paths)} 张图片，已随机抽取其中 {len(images)} 张进行测速和对比。")

images_str = [str(img) for img in images]

# ==========================================
# 4. 模型预热 (Warm-up)
# ==========================================
print("正在进行模型预热 (Warm-up)...")
warmup_img = images_str[0]
for _ in range(3):  
    model1.predict(warmup_img, verbose=False)
    model2.predict(warmup_img, verbose=False)

# 5. 开始正式测速与可视化保存
print(f"开始测试并保存可视化结果...")
times_model1 = []
times_model2 = []
skip_count = 0  # 记录被跳过的图片数量

for i, img_path in enumerate(images_str):
    res1 = model1.predict(img_path, verbose=False)
    res2 = model2.predict(img_path, verbose=False)
    
    # 检查是否检测到了目标，没有检测到则跳过该图
    if len(res1[0].boxes) == 0 or len(res2[0].boxes) == 0:
        skip_count += 1
        continue
    
    # ==================================================
    # ✅ 新增：保存可视化结果
    # ==================================================
    img_stem = Path(img_path).stem   # 获取文件名（不含后缀，例如 'car_001'）
    img_suffix = Path(img_path).suffix # 获取原图后缀（例如 '.jpg'）
    
    # 拼接保存路径
    save_path_baseline = os.path.join(VIS_SAVE_DIR, f"{img_stem}_baseline{img_suffix}")
    save_path_ours = os.path.join(VIS_SAVE_DIR, f"{img_stem}_ours{img_suffix}")
    
    # 调用 YOLO 的内置 save() 方法，直接将带有检测框的图像保存到本地
    res1[0].save(filename=save_path_baseline)
    res2[0].save(filename=save_path_ours)
    # ==================================================

    # 获取纯推理时间 (单位: 毫秒 ms)
    t1 = res1[0].speed['inference']
    t2 = res2[0].speed['inference']
    
    times_model1.append(t1)
    times_model2.append(t2)
    
    if (i + 1) % 10 == 0:
        print(f"进度: {i + 1}/{len(images)}...")

# 6. 检查是否有有效数据
valid_count = len(times_model1)
if valid_count == 0:
    raise ValueError("所有图片都被跳过了（两个模型都没有检测到目标），无法绘制对比图！")

# 7. 计算平均统计数据
avg_t1 = np.mean(times_model1)
avg_t2 = np.mean(times_model2)
fps1 = 1000 / avg_t1 if avg_t1 > 0 else 0
fps2 = 1000 / avg_t2 if avg_t2 > 0 else 0

print("-" * 50)
print("🎯 测试完成！")
print(f"计划处理随机图片: {len(images)} 张")
print(f"未检测到目标被跳过: {skip_count} 张")
print(f"实际参与对比的有效图片: {valid_count} 张")
print(f"📁 可视化对比图片已全部保存至目录: {VIS_SAVE_DIR}/")
print("-" * 50)
print(f"Model 1 (Baseline): 平均 {avg_t1:.2f} ms/图 | 约 {fps1:.1f} FPS")
print(f"Model 2 (Ours):     平均 {avg_t2:.2f} ms/图 | 约 {fps2:.1f} FPS")
print("-" * 50)

# 8. 绘制对比曲线图
plt.figure(figsize=(12, 6))

plt.plot(times_model1, label=f'Model 1 Baseline (Avg: {avg_t1:.2f}ms)', color='blue', alpha=0.8, linewidth=1.5)
plt.plot(times_model2, label=f'Model 2 Ours (Avg: {avg_t2:.2f}ms)', color='red', alpha=0.8, linewidth=1.5)

plt.axhline(y=avg_t1, color='blue', linestyle='--', alpha=0.5)
plt.axhline(y=avg_t2, color='red', linestyle='--', alpha=0.5)

plt.title(f'Inference Speed Comparison ({valid_count} Valid Random Images)', fontsize=14, fontweight='bold')
plt.xlabel('Valid Image Index', fontsize=12)
plt.ylabel('Inference Time (ms)', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
save_path_plot = 'speed_comparison_result.png'
plt.savefig(save_path_plot, dpi=300)
print(f"📊 对比折线图已保存至当前目录: {save_path_plot}")