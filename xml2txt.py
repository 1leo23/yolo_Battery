import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse

path = "./data/xml/labels"  # 填寫自己的位置
classes = {"round": 0,"box":1 }  # 記得在這裡添加所有需要的類別
labels_path = "./data/labels"  # 填寫自己的位置

if not os.path.exists(labels_path):
    os.mkdir(labels_path)  # 如果標籤文件夾不存在，則創建

# train_path = os.path.join(labels_path, "licence")
train_path = labels_path
if not os.path.exists(train_path):
    os.mkdir(train_path)

for annotations in os.listdir(path):
    dom = parse(os.path.join(path, annotations))
    root = dom.documentElement

    # 確保 'filename' 標籤存在並處理
    filename_tag = root.getElementsByTagName("filename")
    if filename_tag:
        filename = filename_tag[0].childNodes[0].data
        filename = ".txt".join(filename.split(".png"))  # 修改檔案擴展名
    else:
        continue  # 如果沒有 'filename' 標籤，跳過

    # 確保 'width' 和 'height' 標籤存在並處理
    image_width_tag = root.getElementsByTagName("width")
    image_height_tag = root.getElementsByTagName("height")
    if image_width_tag and image_height_tag:
        image_width = int(image_width_tag[0].childNodes[0].data)
        image_height = int(image_height_tag[0].childNodes[0].data)
    else:
        continue  # 如果沒有 'width' 或 'height' 標籤，跳過

    # 打開文件寫入
    with open(os.path.join(labels_path, filename), "w") as r:
        for items in root.getElementsByTagName("object"):
            name_tag = items.getElementsByTagName("name")
            if name_tag:
                name = name_tag[0].childNodes[0].data
            else:
                continue  # 如果沒有 'name' 標籤，跳過

            # 驗證 name 是否在 classes 中，否則設為 0（或其他值）
            class_id = classes.get(name, 0)  # 使用 -1 或其他的預設值來處理未知類別

            # 獲取坐標值
            xmin = int(items.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(items.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(items.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(items.getElementsByTagName("ymax")[0].childNodes[0].data)

            # 計算比例
            x_center_norm = ((xmin + xmax) / 2) / image_width
            y_center_norm = ((ymin + ymax) / 2) / image_height
            width_norm = (xmax - xmin) / image_width
            height_norm = (ymax - ymin) / image_height

            # 寫入到文件
            r.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
