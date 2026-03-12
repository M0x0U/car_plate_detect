"""
使用训练好的YOLO车牌检测模型参数, 检测出车牌位置
支持：1. 静态图片批量处理  2. 实时摄像头视频流检测
使用OCR工具进行车牌识别
"""
from ultralytics import YOLO
import cv2
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

MODEL_PARAM_PATH = 'model_params/license_plate_detector.pt'

class OCR_tools:
    # 采用单例模式缓存模型
    _lpr3_catcher = None
    
    @staticmethod
    def pytesseract_predict(img):
        """使用pytesseract OCR工具进行识别"""
        import pytesseract
        PYTESSERACT_OCR_CONFIG = r'--oem 3 --psm 8'
        PYTESSERACT_OCR_TOOL_PATH = r'E:/Tesseract_OCR/tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_OCR_TOOL_PATH
        result = pytesseract.image_to_string(img, config=PYTESSERACT_OCR_CONFIG)
        return result
    
    @staticmethod
    def lpr3_predict(img):
        """使用hyperlpr3进行OCR识别"""
        if OCR_tools._lpr3_catcher is None:
            import hyperlpr3 as lpr3
            OCR_tools._lpr3_catcher = lpr3.LicensePlateCatcher()
        results = OCR_tools._lpr3_catcher(img)
        if results:
            return results[0][0]
        return ""

def put_chinese_text(img, text, pos, font_size=30, color=(0,255,0)):
    """在图片上写入中文字符"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        try:
            font = ImageFont.truetype("Arial Unicode.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    draw.text(pos, text, fill=color, font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def recognize_plate_ocr(img, ocr_engine='lpr3'):
    """预处理 → OCR识别 → 输出车牌字符"""
    if ocr_engine == 'tesseract':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = OCR_tools.pytesseract_predict(processed_img)
    else:
        result = OCR_tools.lpr3_predict(img)

    if result:
        plate = result.strip().replace(" ", "").replace("\n", "")
    else:
        plate = 'unknown'
    return plate

def check(plate_crop, plate_num):
    """手动矫正车牌 (仅用于图片模式)"""
    try:
        print(f"🔍 OCR 自动识别结果：[{plate_num}]")
        cv2.imshow("【请手动校正车牌】按任意键继续", plate_crop)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        user_input = input("✏️ 请输入正确车牌（直接回车使用OCR结果）：").strip()
        
        if user_input:
            final_plate = user_input
        else:
            final_plate = plate_num
            
        if final_plate.lower() == 'false':
            return False, plate_num
            
        return True, final_plate
    except:
        return False, plate_num

def run_camera_stream(model, args):
    """
    打开摄像头进行实时车牌检测与识别（极致流畅优化版）
    """
    cap = cv2.VideoCapture(args.camera_id)
    
    # ⚡ 优化1：限制摄像头读取分辨率，极大降低运算压力
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"❌ 错误：无法打开摄像头 ID: {args.camera_id}")
        return

    print(f"🟢 摄像头已开启！使用的OCR引擎: {args.ocr}")
    print("👉 按下键盘上的 'q' 键退出实时检测。")

    frame_count = 0
    plate_cache = []  # 用于缓存上一帧的车牌信息：[{'center': (cx, cy), 'text': '粤A12345'}]

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # ⚡ 优化2：降低 YOLO 推理分辨率 (imgsz=480)，速度可提升 1.5 倍
        results = model.predict(frame, imgsz=480, verbose=False)
        frame_draw = frame.copy()
        
        current_cache = [] # 记录当前帧的所有车牌，用于传给下一帧

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            
            # 边界保护
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 计算车牌框的中心点
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            plate_num = ""
            need_ocr = True

            # ⚡ 优化3：智能 OCR 抽帧缓存策略
            # 如果不是第5帧的倍数，尝试从上一帧的缓存中“偷懒”拿结果
            if frame_count % 10 != 0:
                for cached in plate_cache:
                    # 计算当前框中心点和上一帧框中心点的距离
                    dist = (cx - cached['center'][0])**2 + (cy - cached['center'][1])**2
                    if dist < 15000:  # 如果距离很近，说明是同一个车牌
                        plate_num = cached['text']
                        need_ocr = False
                        break  # 找到了就不用做 OCR 了！

            # 如果需要 OCR（比如刚好第5帧，或者画面里突然出现了一个新车牌）
            if need_ocr:
                plate_crop = frame[y1:y2, x1:x2]  
                if plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                    plate_num = recognize_plate_ocr(plate_crop, ocr_engine=args.ocr)
            
            # 把当前车牌信息存入缓存，留给下一帧用
            current_cache.append({'center': (cx, cy), 'text': plate_num})

            # 实时画框和文字
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_num and plate_num != 'unknown':
                frame_draw = put_chinese_text(frame_draw, plate_num, (x1, max(0, y1 - 35)), font_size=32, color=(0,255,0))
            
        # 更新全局缓存
        plate_cache = current_cache

        # 显示实时画面
        cv2.imshow("Real-time License Plate Detection", frame_draw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 已手动停止实时检测。")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 释放摄像头资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


def process_images(model, args):
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in img_exts:
        img_paths.extend(list(Path(args.source).glob(ext)))

    if not img_paths:
        print(f"❌ 错误：在 {args.source} 中未找到任何图片！")
        return
    
    total_found = len(img_paths)
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
        print(f"共找到 {total_found} 张图片，将处理前 {len(img_paths)} 张。")
    else:
        print(f"共找到 {len(img_paths)} 张图片，开始处理...")

    save_dir = Path(args.result)
    save_img_dir = save_dir / "detected" 
    save_crop_dir = save_dir / "crops" 
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_crop_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.result) / args.csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名', '车牌序号', '识别结果', '是否手动修正'])
        
    for img_path in img_paths:
        img_name = img_path.stem 
        print(f'Processing {img_name}')
        img_path_str = str(img_path)

        results = model.predict(img_path_str, verbose=False)
        img = cv2.imread(img_path_str)
        img_draw = img.copy()
        
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            plate_crop = img[y1:y2, x1:x2]  
            plate_num = recognize_plate_ocr(plate_crop, ocr_engine=args.ocr)
            
            if args.manual:
                plate_check, plate_num = check(plate_crop, plate_num)
                if not plate_check: continue
                manually_corrected = True
            else:
                manually_corrected = False
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([img_name, idx, plate_num, '是' if manually_corrected else '否'])
            
            crop_save_path = save_crop_dir / f"{img_name}_{idx}.png"
            cv2.imwrite(str(crop_save_path), plate_crop)
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_draw = put_chinese_text(img_draw, plate_num, (x1, max(0, y1 - 35)), font_size=32, color=(0,255,0))
        
        draw_save_path = save_img_dir / f"{img_name}.png"
        cv2.imwrite(str(draw_save_path), img_draw)


def main():
    parser = argparse.ArgumentParser(description='车牌检测系统 (支持图片与实时摄像头)')
    
    # 互斥参数组：要么处理图片，要么处理视频流
    parser.add_argument('--camera_id', type=int, default=-1, help='开启实时摄像头检测，输入摄像头ID (默认0为内置摄像头)')
    
    # 图片处理参数
    parser.add_argument('--source', type=str, default='images/', help='输入图片文件夹路径')
    parser.add_argument('--result', type=str, default='results/', help='输出结果根目录')
    parser.add_argument('--manual', action='store_true', help='启用手动校正模式（仅限图片模式）')
    parser.add_argument('--csv', type=str, default='results.csv', help='保存识别结果的CSV文件')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的图片数量')
    
    # OCR 参数
    parser.add_argument('--ocr', type=str, default='lpr3', choices=['lpr3', 'tesseract'], help='选择OCR引擎')
    
    args = parser.parse_args()
    
    print("⏳ 正在加载 YOLO 模型...")
    model = YOLO(MODEL_PARAM_PATH)
    
    # 根据参数决定运行模式
    if args.camera_id >= 0:
        # 运行实时摄像头模式
        run_camera_stream(model, args)
    else:
        # 运行静态图片模式
        process_images(model, args)
    
if __name__ == '__main__':
    main()