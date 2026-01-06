import os
import cv2
from ultralytics import YOLO

# 確保模型檔案存在
model_path = "D:/AOI-project2/runs/detect/train/weights/best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)  # 載入訓練好的模型
else:
    print(f"模型檔案 {model_path} 不存在，請檢查路徑！")
    exit(1)

# 資料夾路徑
input_folder = './vedio'  # 輸入資料夾，包含需要檢測的影片
output_folder = './out_vedio'  # 輸出資料夾，儲存標註結果的影片

# 如果輸出資料夾不存在，則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 設置正常和不正常的顏色
normal_color = (0, 255, 0)  # 綠色
abnormal_color = (0, 0, 255)  # 紅色

# 進行影片辨識與處理
for video_name in os.listdir(input_folder):
    if video_name.lower().endswith(('.mp4', '.avi', '.mov')):  # 檢查是否是影片
        video_path = os.path.join(input_folder, video_name)
        cap = cv2.VideoCapture(video_path)

        # 獲取影片的基本資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 輸出格式
        output_video_path = os.path.join(output_folder, video_name)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 影片結束

            # 使用 YOLO 模型進行預測
            results = model(frame)

            if isinstance(results, list):
                results = results[0]  # 如果是 list，取第一個元素作為結果

            # 用來存儲 round 重心的位置
            round_centers = []

            # 取得檢測框資料
            for result in results.boxes:
                xmin, ymin, xmax, ymax = result.xyxy[0].tolist()  # 取得座標
                class_id = int(result.cls)  # 類別 ID
                confidence = result.conf[0].item()  # 信心度

                if class_id == 0:  # 類別 round
                    # 繪製紅色框
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), abnormal_color, 2)

                    # 計算重心座標
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    round_centers.append((center_x, center_y))
                    # 標示重心座標
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)  # 白色圓圈
                    cv2.putText(frame, f"({int(center_x)}, {int(center_y)})",
                                (int(center_x) + 10, (int(center_y) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    # 繪製其他類別的框
                    color = normal_color
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

            # 根據 X 座標對 round_centers 進行排序
            round_centers = sorted(round_centers, key=lambda x: x[0])  # 按 X 座標大小排序

            # 確保有至少 3 個座標
            if len(round_centers) >= 3:
                second_x = round_centers[1][0]
                third_x = round_centers[2][0]
                diff = third_x - second_x

                if diff < 20:
                    label = "good"
                    label_color = normal_color
                else:
                    label = "error"
                    label_color = abnormal_color

                cv2.putText(frame, label, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

            # 判斷 round 數量是否少於 5
            if len(round_centers) < 5:
                label = "error"
                label_color = abnormal_color
                cv2.putText(frame, label, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

            # 寫入處理後的影格
            out.write(frame)

        cap.release()
        out.release()
        print(f"處理完成: {video_name}")

print("所有影片已處理完成！")
