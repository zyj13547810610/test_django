import os
import shutil
import xml.etree.ElementTree as ET
import tqdm
import yaml
import random

class Dataset_Process:
    def __init__(self):
        pass

    def parse_xml_label(self, xml_path):
        """解析VOC格式的xml文件，返回所有标注信息"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        size = root.find('size')
        if size is not None:
            width_elem = size.find('width')
            height_elem = size.find('height')
            try:
                img_w = int(width_elem.text) if width_elem is not None and width_elem.text is not None else 1
            except Exception:
                img_w = 1
            try:
                img_h = int(height_elem.text) if height_elem is not None and height_elem.text is not None else 1
            except Exception:
                img_h = 1
        else:
            img_w = img_h = 1  # 防止除零
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            name = name_elem.text if name_elem is not None and name_elem.text is not None else "unknown"
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    xmin = float(xmin_elem.text) if xmin_elem is not None and xmin_elem.text is not None else 0.0
                    ymin = float(ymin_elem.text) if ymin_elem is not None and ymin_elem.text is not None else 0.0
                    xmax = float(xmax_elem.text) if xmax_elem is not None and xmax_elem.text is not None else 0.0
                    ymax = float(ymax_elem.text) if ymax_elem is not None and ymax_elem.text is not None else 0.0
                except Exception:
                    xmin = ymin = xmax = ymax = 0.0
                objects.append({
                    'name': name,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'img_w': img_w,
                    'img_h': img_h
                })
        return objects

    def convert_bbox_to_yolo(self, bbox, img_w, img_h):
        """将VOC的bbox转换为YOLO格式"""
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        return [x_center, y_center, w, h]

    def get_classes_from_dir(self, xml_dir):
        """扫描所有xml，收集所有类别"""
        classes = set()
        for fname in os.listdir(xml_dir):
            if fname.endswith('.xml'):
                xml_path = os.path.join(xml_dir, fname)
                objects = self.parse_xml_label(xml_path)
                for obj in objects:
                    classes.add(obj['name'])
        return sorted(list(classes))

    def voc2yolo(self, xml_dir, save_label_dir, save_img_dir, classes, file_names=None,img_src_dir=None):
        """将xml目录下所有标注转为YOLO格式txt，并复制图片"""
        if file_names is not None:
            xml_list = file_names
        else:
            xml_list = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        for fname in tqdm.tqdm(xml_list,desc=f"{os.path.basename(save_label_dir)}"):
            if not fname.endswith('.xml'):
                continue
            xml_path = os.path.join(xml_dir, fname)
            objects = self.parse_xml_label(xml_path)
            if not objects:
                continue
            txt_name = os.path.splitext(fname)[0] + '.txt'
            txt_path = os.path.join(save_label_dir, txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for obj in objects:
                    if obj['name'] not in classes:
                        continue
                    class_id = classes.index(obj['name'])
                    yolo_bbox = self.convert_bbox_to_yolo(obj['bbox'], obj['img_w'], obj['img_h'])
                    yolo_bbox = [str(round(x, 6)) for x in yolo_bbox]
                    f.write(f"{class_id} {' '.join(yolo_bbox)}\n")
            # 复制图片到目标images目录
            img_name = os.path.splitext(fname)[0] + '.jpg'
            # 支持jpg和png
            found_img = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = os.path.join(img_src_dir if img_src_dir is not None else xml_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(img_file):
                    shutil.copy(img_file, os.path.join(save_img_dir, os.path.basename(img_file)))
                    found_img = True
                    break
            if not found_img:
                print(f"警告: 未找到图片 {img_name}，已跳过。")

    def generate_yaml(self, base_processed_dir, classes):
        """
        生成数据集的yaml配置文件
        """
        yaml_dict = {}
        yaml_dict['path'] = base_processed_dir
        yaml_dict['train'] = 'images/train'
        yaml_dict['val'] = 'images/val'
        yaml_dict['test'] = ''  # 可选
        yaml_dict['names'] = {i: c for i, c in enumerate(classes)}
        yaml_path = os.path.join(base_processed_dir, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)
        print(f"已生成数据集配置文件: {yaml_path}")

    def detect_dataset_format(self, base_dir):
        """
        检测数据集格式，返回 'split' 或 'flat'
        split: 有train/val子文件夹
        flat: 只有一个文件夹，直接包含图片和xml
        """
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if 'train' in subdirs and 'val' in subdirs:
            return 'split'
        # 检查是否有大量图片和xml
        xmls = [f for f in os.listdir(base_dir) if f.endswith('.xml')]
        imgs = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(xmls) > 0 and len(imgs) > 0:
            return 'flat'
        raise Exception("无法识别数据集格式，请检查目录结构。")

    def process_flat(self, base_dir, train_ratio=0.8, seed=42):
        """
        处理只有一个文件夹的情况（flat格式），自动划分train/val，并调用voc2yolo进行格式转换
        """
        print("检测到数据集为单文件夹格式（flat），自动划分训练集和验证集。")
        base_processed_dir = os.path.join(base_dir, 'precessed')
        images_dir = os.path.join(base_processed_dir, 'images')
        labels_dir = os.path.join(base_processed_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        train_img_dir = os.path.join(images_dir, 'train')
        val_img_dir = os.path.join(images_dir, 'val')
        train_label_dir = os.path.join(labels_dir, 'train')
        val_label_dir = os.path.join(labels_dir, 'val')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # 获取所有xml文件名（不带扩展名）
        xml_files = [f for f in os.listdir(base_dir) if f.endswith('.xml')]
        base_names = [os.path.splitext(f)[0] for f in xml_files]
        # 随机划分
        random.seed(seed)
        random.shuffle(base_names)
        n_train = int(len(base_names) * train_ratio)
        train_names = base_names[:n_train]
        val_names = base_names[n_train:]

        # 收集类别
        classes = set()
        for name in train_names:
            xml_path = os.path.join(base_dir, name + '.xml')
            objects = self.parse_xml_label(xml_path)
            for obj in objects:
                classes.add(obj['name'])
        classes = sorted(list(classes))
        print("训练数据集标签类别:", classes)
        # 保存类别到文件
        with open(os.path.join(base_processed_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in classes:
                f.write(f"{c}\n")

        # 用voc2yolo处理train和val
        self.voc2yolo(
            xml_dir=base_dir,
            save_label_dir=train_label_dir,
            save_img_dir=train_img_dir,
            classes=classes,
            file_names=[name + '.xml' for name in train_names],
            img_src_dir=base_dir
        )
        self.voc2yolo(
            xml_dir=base_dir,
            save_label_dir=val_label_dir,
            save_img_dir=val_img_dir,
            classes=classes,
            file_names=[name + '.xml' for name in val_names],
            img_src_dir=base_dir
        )

        # 生成yaml配置
        self.generate_yaml(base_processed_dir, classes)
        print("转换完成，YOLO标签和图片已生成在 processed_dataset。")

    def process_split(self, base_dir):
        """
        处理有train/val子文件夹的情况（split格式）
        """
        print("检测到数据集为train/val子文件夹格式（split）。")
        base_processed_dir = os.path.join(base_dir, 'precessed')
        images_dir = os.path.join(base_processed_dir, 'images')
        labels_dir = os.path.join(base_processed_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for split in ['train', 'val']:
            xml_dir = os.path.join(base_dir, split, 'labels') if os.path.exists(os.path.join(base_dir, split, 'labels')) else os.path.join(base_dir, split)
            img_src_dir = os.path.join(base_dir, split, 'images') if os.path.exists(os.path.join(base_dir, split, 'images')) else os.path.join(base_dir, split)
            save_img_dir = os.path.join(images_dir, split)
            save_label_dir = os.path.join(labels_dir, split)
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_label_dir, exist_ok=True)
            # 先收集所有类别
            if split == 'train':
                classes = self.get_classes_from_dir(xml_dir)
                print("训练数据集标签类别:", classes)
                # 保存类别到文件
                with open(os.path.join(base_processed_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
                    for c in classes:
                        f.write(f"{c}\n")
            else:
                # 验证集用训练集的类别
                with open(os.path.join(base_processed_dir, 'classes.txt'), 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f.readlines()]
            self.voc2yolo(xml_dir, save_label_dir, save_img_dir, classes, img_src_dir=img_src_dir)
            
        print("转换完成，YOLO标签和图片已生成在 processed_dataset。")
        self.generate_yaml(base_processed_dir, classes)

    def process(self, base_dir):
        """
        主流程：自动检测数据集格式并处理
        """
        fmt = self.detect_dataset_format(base_dir)
        if fmt == 'split':
            self.process_split(base_dir)
        elif fmt == 'flat':
            self.process_flat(base_dir)
        else:
            raise Exception("未知数据集格式，无法处理。")

# 主流程入口
if __name__ == "__main__":
    # base_dir='my_dataset/test'
    base_dir = 'my_dataset/phone_small'
    dataset_process = Dataset_Process()
    dataset_process.process(base_dir)
