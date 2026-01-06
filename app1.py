import os
import cv2
from ultralytics import YOLO
import numpy as np

# 確保模型檔案存在
model_path = r"C:\Users\user1\Desktop\yolo\Battery\runs\detect\train\weights\best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)  # 載入訓練好的模型
else:
    print(f"模型檔案 {model_path} 不存在，請檢查路徑！")
    exit(1)

# 資料夾路徑
input_folder = './images'  # 輸入資料夾，包含需要檢測的圖片
output_folder = './out_images'  # 輸出資料夾，儲存標註結果的圖片

# 如果輸出資料夾不存在，則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 設置正常和不正常的顏色
normal_color = (0, 255, 0)  # 綠色
abnormal_color = (0, 0, 255)  # 紅色

# 進行圖片辨識與處理
for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 檢查是否是圖片
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        # 使用 YOLO 模型進行預測
        results = model(image_path)

        # 這裡確認 results 是一個 list
        if isinstance(results, list):
            results = results[0]  # 如果是 list，取第一個元素作為結果

        # 用來存儲 round 重心的位置
        round_centers = []
        boxes = []  # 儲存所有 box 類別的位置

        # 取得檢測框資料
        for result in results.boxes:
            xmin, ymin, xmax, ymax = result.xyxy[0].tolist()  # 取得座標
            class_id = int(result.cls)  # 類別 ID
            confidence = result.conf[0].item()  # 信心度

            # 類別名稱，根據您的訓練類別設置
            if class_id == 0:  # 類別 round
                color = (0, 0, 255)  # 紅色框
                # 計算重心座標
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                # 標示重心座標
                round_centers.append((center_x, center_y))
                cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)  # 白色圓圈
                cv2.putText(image, f"({int(center_x)}, {int(center_y)})", (int(center_x) + 10, (int(center_y) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif class_id == 1:  # 類別 box
                color = (0, 255, 0)  # 綠色框
                boxes.append((xmin, ymin, xmax, ymax))
            else:
                continue  # 忽略不需要的類別

            # 畫出邊界框
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

            # 可以選擇加上類別名稱與信心度標籤
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(image, label, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 根據 X 座標對 round_centers 進行排序
        round_centers = sorted(round_centers, key=lambda x: x[0])  # 按 X 座標大小排序

        # 確保有至少 3 個座標
        if len(round_centers) >= 3:
            # 取第二和第三個 X 座標進行比較
            second_x = round_centers[1][0]  # 排序後的第二個
            third_x = round_centers[2][0]  # 排序後的第三個

            # 計算差值
            diff = third_x - second_x

            # 判斷差值是否小於 20
            if diff < 20:
                label = "good"  # 正常
                label_color = normal_color
            else:
                label = "error"  # 不正常
                label_color = abnormal_color

            # 在圖片右上角標註 "正常" 或 "不正常"
            cv2.putText(image, label, (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        # 判斷 round 數量是否少於 5
        if len(round_centers) < 5:
            label = "error"
            label_color = abnormal_color
            cv2.putText(image, label, (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        # 儲存處理過的圖片
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image)

        print(f"處理完成: {image_name}")

print("所有圖片已處理完成！")
