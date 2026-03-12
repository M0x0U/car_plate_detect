"""
使用训练好的YOLO车牌检测模型参数, 检测出车牌位置
使用OCR工具进行车牌识别
保存结果图片
"""
from ultralytics import YOLO
import cv2
import argparse
import os
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np



MODEL_PARAM_PATH = 'model_params/license_plate_detector.pt'

class OCR_tools:
    # 每个OCR工具封装成方法
    _lpr3_catcher = None
    
    @staticmethod
    def pytessract_predict(img):
        """
        使用pytessract OCR工具进行识别
        """
        import pytesseract
        PYTESSERACT_OCR_CONFIG = r'--oem 3 --psm 8'
        PYTESSERACT_OCR_TOOL_PATH = r'E:/Tesseract_OCR/tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_OCR_TOOL_PATH
        result = pytesseract.image_to_string(img, config=PYTESSERACT_OCR_CONFIG)
        # 此时result是识别的车牌号, 纯字符串
        return result
    
    @staticmethod
    def lpr3_predict(img):
        """
        使用hyperlpr3进行OCR识别
        """
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
    """
    读取车牌图片 → 预处理（增强识别率）→ OCR识别 → 输出车牌字符
    根据args中OCR类别使用对应的OCR识别工具
    """
    if ocr_engine == 'tesseract':
        #* Tesseract预处理：灰度 + 二值化（车牌识别专用，大幅提升准确率）
        # 预处理过程单独封装成一个函数
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = OCR_tools.pytesseract_predict(processed_img)
    else:
        result = OCR_tools.lpr3_predict(img)

    # 清洗结果：去掉空格、换行、无用符号
    if result:
        plate = result.strip().replace(" ", "").replace("\n", "")
    else:
        plate = 'unknown'
    return plate

def check(plate_crop, plate_num):
    # trick: 手动矫正一下车牌
    # 显示图像
    try:
        print(f"🔍 OCR 自动识别结果：[{plate_num}]")
        cv2.imshow("【请手动校正车牌】按任意键继续", plate_crop)
        cv2.waitKey(0)  # 等待按任意键关闭窗口
        cv2.destroyAllWindows()
        user_input = input("✏️ 请输入正确车牌（直接回车使用OCR结果）：").strip()
        # 4. 确定最终结果
        if user_input:
            final_plate = user_input
        else:
            final_plate = plate_num
            
        if final_plate == 'false':
            return False, plate_num
            
        return True, final_plate
    
    except:
        return False, plate_num


def main():
    parser = argparse.ArgumentParser(description='车牌检测批量处理')
    parser.add_argument('--source', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--result', type=str, required=True, help='输出结果根目录')
    parser.add_argument('--manual', action='store_true', help='启用手动校正模式（默认关闭）')
    parser.add_argument('--csv', type=str, default='results.csv', help='保存识别结果的CSV文件路径')
    parser.add_argument('--ocr', type=str, default='lpr3', choices=['lpr3', 'tesseract'], help='选择OCR引擎')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的图片数量')
    args = parser.parse_args()
    
    # 模型加载
    model = YOLO(MODEL_PARAM_PATH)  # 加载模型参数
    
    # 获取 source 文件夹下所有图片文件（支持 jpg/png/jpeg）
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in img_exts:
        img_paths.extend(list(Path(args.source).glob(ext)))

    if not img_paths:
        print(f"错误：在 {args.source} 中未找到任何图片！")
        return
    total_found = len(img_paths)
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
        print(f"共找到 {total_found} 张图片，根据 --limit {args.limit} 将处理前 {len(img_paths)} 张。")
    else:
        print(f"共找到 {len(img_paths)} 张图片，开始处理...")

    
    # 创建输出文件夹
    save_dir = Path(args.result)
    save_img_dir = save_dir / "detected"  # 画框后的图片存在这里
    save_crop_dir = save_dir / "crops"  # 裁剪后的车牌存在这里
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_crop_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备CSV文件记录识别结果
    csv_path = Path(args.result) / args.csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名', '车牌序号', '识别结果', '是否手动修正'])
        
    
    for img_path in img_paths:
        img_name = img_path.stem  # 获取文件名（不含后缀）
        print(f'Processing {img_name}')
        img_path_str = str(img_path)

        # 推理
        results = model.predict(img_path_str)
        
        img = cv2.imread(img_path_str)
        img_draw = img.copy()
        
        # 遍历所有检测到的车牌框(图中可能有多个车牌)
        for idx, box in enumerate(results[0].boxes):
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 

            # 裁剪车牌区域 
            plate_crop = img[y1:y2, x1:x2]  
            # 对裁剪的车牌进行OCR识别
            plate_num = recognize_plate_ocr(plate_crop, ocr_engine=args.ocr)
            
            # 根据参数决定是否手动校正
            if args.manual:
                plate_check, plate_num = check(plate_crop, plate_num, interactive=True)
                if not plate_check:
                    continue
                manually_corrected = True
            else:
                plate_check = True
                manually_corrected = False
            
            # 记录到CSV
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([img_name, idx, plate_num, '是' if manually_corrected else '否'])
            
            # 保存裁剪的车牌 
            crop_save_path = save_crop_dir / f"{img_name}_{idx}.png"
            cv2.imwrite(str(crop_save_path), plate_crop)
            
            # 在原图上标注识别的车牌号码
            # 1. 画出车牌方框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 2. 显示车牌号
            # ✅ 支持【中文+字母+数字】的车牌显示
            img_draw = put_chinese_text(img_draw, plate_num, (x1, y1 - 35), font_size=32, color=(0,255,0))
        
        # 保存标注车牌号后的整张图
        draw_save_path = save_img_dir / f"{img_name}.png"
        cv2.imwrite(str(draw_save_path), img_draw)
        
    
if __name__ == '__main__':
    main()