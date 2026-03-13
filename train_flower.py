from ultralytics import YOLO
import os
import warnings

# 忽略无关警告，让输出更整洁
warnings.filterwarnings('ignore')


def train_yolov8_classifier():
	# ====================== 核心参数配置 ======================
	model_name = "yolov8n-cls.pt"
	data_path = "./datasets/flowers"
	img_size = 224
	epochs = 50
	batch_size = 16
	lr0 = 0.01
	device = 0  # 确保已重装适配的 PyTorch 版本
	save_dir = "./runs/classify"

	# ====================== 环境变量适配（RTX 5060）======================
	import os
	os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
	os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

	# ====================== 检查路径 + 加载模型 ======================
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"数据集路径不存在：{data_path}")

	model = YOLO(model_name)

	# ====================== 开始训练（无警告版）======================
	print("\n开始训练 YOLOv8 分类模型（无增强警告）...")
	results = model.train(
		data=data_path,
		epochs=epochs,
		imgsz=img_size,
		batch=batch_size,
		lr0=lr0,
		device=device,
		project=save_dir,
		name="train_exp",
		exist_ok=True,
		patience=10,
		save=True,
		val=True,
		auto_augment="randaugment",  # 分类模型专用增强
		verbose=True
	)

	# ====================== 验证模型（无警告）======================
	print("\n训练完成！开始评估模型性能...")
	metrics = model.val(
		data=data_path,
		augment=False,  # 显式关闭增强
		imgsz=img_size
	)
	print(f"验证集 TOP1 准确率：{metrics.top1:.4f}")
	print(f"验证集 TOP5 准确率：{metrics.top5:.4f}")

	final_model_path = os.path.join(save_dir, "train_exp", "weights", "best.pt")
	print(f"\n最佳模型已保存至：{final_model_path}")
	return results, metrics


if __name__ == "__main__":
	try:
		# 执行训练函数
		train_results, val_metrics = train_yolov8_classifier()
	except Exception as e:
		print(f"\n训练过程中出错：{e}")
		exit(1)