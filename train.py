from ultralytics import YOLO

# 加载 YOLOv8n 预训练模型
model = YOLO("model_params/yolov8n.pt")

# 开始训练
model.train(
    data="config/plate.yaml",    # 你的配置文件
    epochs=100,           # 训练轮数（100够了）
    batch=8,              # 批次大小（显存小就改 4 或 2）
    imgsz=640,            # 图片尺寸
    device="0",         # 如果你有GPU就写 0，没有就写 cpu
    name="plate_detector" # 模型保存名称
)