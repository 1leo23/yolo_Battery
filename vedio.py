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
input_folder = './video'  # 輸入資料夾，包含需要檢測的影片
output_folder = './out_video'  # 輸出資料夾，儲存標註結果的影片

# 如果輸出資料夾不存在，則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 設置顏色
round_color = (0, 0, 255)  # 紅色

# 進行影片辨識與處理
for video_name in os.listdir(input_folder):
    if video_name.lower().endswith(('.mp4', '.avi', '.mov')):  # 檢查是否是影片
        video_path = os.path.join(input_folder, video_name)
        output_video_path = os.path.join(output_folder, f"processed_{video_name}")

        # 讀取影片
        cap = cv2.VideoCapture(video_path)

        # 檢查影片是否成功開啟
        if not cap.isOpened():
            print(f"無法開啟影片檔案 {video_name}")
            continue

        # 影片的寬、高和FPS設定
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 輸出影片的設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用於輸出的影片格式
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # 讀取每一幀並處理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 使用 YOLO 模型進行預測
            results = model(frame)

            # 這裡確認 results 是一個 list
            if isinstance(results, list):
                results = results[0]  # 如果是 list，取第一個元素作為結果

            # 用來存儲 round 重心的位置
            round_centers = []

            # 取得檢測框資料
            for result in results.boxes:
                xmin, ymin, xmax, ymax = result.xyxy[0].tolist()  # 取得座標
                class_id = int(result.cls)  # 類別 ID

                # 如果是類別 round（假設類別 ID 是 0）
                if class_id == 0:
                    # 計算重心座標
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    round_centers.append((center_x, center_y))

                    # 標示重心座標
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)  # 白色圓圈
                    cv2.putText(frame, f"({int(center_x)}, {int(center_y)})", (int(center_x) + 10, int(center_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 畫出紅色邊界框
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), round_color, 2)

            # 將處理過的幀寫入輸出影片
            out.write(frame)

        # 釋放影片資源
        cap.release()
        out.release()

        print(f"處理完成: {video_name}")

print("所有影片已處理完成！")
