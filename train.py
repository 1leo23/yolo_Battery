import os
import shutil
import time
from ultralytics import YOLO

if __name__ == '__main__':
    train_path = "./runs/train"

    # 刪除舊的訓練結果目錄`
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    # 載入 YOLOv11 預訓練模型
    model = YOLO("yolo11s.pt")
    print("開始訓練 .........")

    t1 = time.time()

    # 開始訓練
    model.train(
        data=r"C:\Users\user1\Desktop\yolo\Battery\data.yaml",
        epochs=200,          # 小資料集不用 300
        imgsz=416,
        batch=32,            # RTX 4070 OK
        device=0,            # ★ 明確指定 GPU
        workers=4,           # ★ 非常重要（Windows 不要太高）
        amp=True             # ★ 混合精度，GPU 會快很多
)git add

    t2 = time.time()
    print(f'訓練花費時間 : {t2 - t1}秒')

    # 匯出模型
    path = model.export()
    print(f'模型匯出路徑 : {path}')

