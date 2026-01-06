import os
import random
import shutil
import stat  # 必須導入這個庫來處理權限

data_path = './data'               # 填寫自己的位置
train_path = './data/train'        # 填寫自己的位置
valid_path = './data/valid'        # 填寫自己的位置

# --- 關鍵修復函數：強制解除唯讀屬性 ---
def handle_remove_readonly(func, path, exc):
    """
    當 shutil.rmtree 遇到權限錯誤時，執行此函數。
    它會修改檔案權限為「可寫入」，然後再次嘗試刪除。
    """
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == 13: # errno 13 是權限錯誤
        os.chmod(path, stat.S_IWRITE) # 強制改為可寫
        func(path) # 重試刪除
    else:
        raise # 如果是其他嚴重錯誤，則拋出異常

# --- 主流程 ---

# 1. 清理目錄 (加入 onerror 參數來處理權限問題)
print("正在清理舊目錄...")
if os.path.exists(train_path):
    shutil.rmtree(train_path, onerror=handle_remove_readonly)
if os.path.exists(valid_path):
    shutil.rmtree(valid_path, onerror=handle_remove_readonly)

# 2. 創建目錄
os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'labels'), exist_ok=True)

# 3. 獲取所有文件
print("正在讀取檔案列表...")
files = [os.path.splitext(file)[0]
         for file in os.listdir(os.path.join(data_path, "images"))
         if file.endswith('.png') or file.endswith('.jpg')]
random.shuffle(files)

# 4. 分割數據
mid = int(len(files) * 0.8)
print(f"總共 {len(files)} 張圖片。訓練集: {mid}, 驗證集: {len(files)-mid}")

# 處理訓練集
print("正在處理訓練集...")
for file in files[:mid]:
    for ext in ['.png', '.jpg']:
        source = os.path.join(data_path, "images", f'{file}{ext}')
        target = os.path.join(train_path, "images", f'{file}{ext}')
        if os.path.exists(source):
            shutil.copy(source, target)

    source = os.path.join(data_path, "labels", f'{file}.txt')
    target = os.path.join(train_path, "labels", f'{file}.txt')
    if os.path.exists(source):
        shutil.copy(source, target)

# 處理驗證集
print("正在處理驗證集...")
for file in files[mid:]:
    for ext in ['.png', '.jpg']:
        source = os.path.join(data_path, "images", f'{file}{ext}')
        target = os.path.join(valid_path, "images", f'{file}{ext}')
        if os.path.exists(source):
            shutil.copy(source, target)

    source = os.path.join(data_path, "labels", f'{file}.txt')
    target = os.path.join(valid_path, "labels", f'{file}.txt')
    if os.path.exists(source):
        shutil.copy(source, target)

print("完成！")
