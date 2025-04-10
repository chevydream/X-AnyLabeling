import base64
import json
import os
import os.path as osp
from PIL import Image
import time
import numpy as np
import cv2
import pathlib
import shutil
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.label_file import io_open
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import get_msg_box_style
from anylabeling.views.labeling.widgets.popup import Popup


__all__ = ["run_all_images"]


INVALID_MODEL_LIST = [
    "segment_anything",
    "segment_anything_2",
    "sam_med2d",
    "sam_hq",
    "efficientvit_sam",
    "edge_sam",
    "open_vision",
    "geco",
]

TEXT_PROMPT_MODELS = [
    "grounding_dino",
    "grounding_sam",
    "grounding_sam2",
]

VIDEO_MODELS = [
    "segment_anything_2_video",
]


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            "font-size: 13px; color: #1d1d1f; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
            
            QLineEdit {
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                background-color: #F9F9F9;
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
            }
            
            QLineEdit:hover {
                background-color: #DBDBDB;
            }
            
            QLineEdit:focus {
                border: 2px solid #0066FF;
                background-color: #F9F9F9;
            }
            
            QPushButton {
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }
            
            QPushButton[text="OK"] {
                background-color: #0066FF;
                color: white;
                border: none;
            }
            
            QPushButton[text="OK"]:hover {
                background-color: #0077ED;
            }
            
            QPushButton[text="OK"]:pressed {
                background-color: #0068D0;
            }
            
            QPushButton[text="Cancel"] {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
            }
            
            QPushButton[text="Cancel"]:hover {
                background-color: #e5e5e5;
            }
            
            QPushButton[text="Cancel"]:pressed {
                background-color: #d5d5d5;
            }
        """
        )

    def get_input_text(self):
        if self.exec_() == QDialog.Accepted:
            return self.text_input.text().strip()
        return ""


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def finish_processing(self, progress_dialog):
    self.filename = self.image_list[self.current_index]
    self.import_image_folder(osp.dirname(self.filename))

    del self.text_prompt
    del self.run_tracker
    del self.image_index
    del self.current_index

    progress_dialog.close()

    popup = Popup(
        self.tr("Processing completed successfully!"),
        self,
        icon="anylabeling/resources/icons/copy-green.svg",
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        # 移除归并类
        wgb_shapes = []
        for shape in auto_labeling_result.shapes:
            if shape.label.find("(gbl)") == -1:
                wgb_shapes.append(shape)

        new_shapes = [shape.to_dict() for shape in wgb_shapes]

        new_description = auto_labeling_result.description
        replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                data["shapes"].extend(new_shapes)
                data["description"] += new_description
        else:
            if self._config["store_data"]:
                image_data = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data = None

            image_path = osp.basename(image_file)
            image_width, image_height = get_image_size(image_file)

            data = {
                "version": __version__,
                "flags": {},
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
            }

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file {image_file}: {str(e)}"
        )

def crop_save_zitu(self, image_file, label, points, score, flagStr=None):
    image_path = pathlib.Path(image_file)
    orig_filename = image_path.stem

    # Read image safely handling non-ASCII paths
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    # Calculate crop coordinates
    xmin = int(points[0].x())
    ymin = int(points[0].y())
    xmax = int(points[2].x())
    ymax = int(points[2].y())

    # Crop image with bounds checking
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(width, xmax), min(height, ymax)
    crop_image = image[ymin:ymax, xmin:xmax]
    # Create output directory
    subPath = f"0.{int(10 * score)}~0.{int(10 * (score + 0.1))}" if int(10 * score) < 9 else f"0.9~1.0"
    if flagStr is None:
        dst_path = pathlib.Path(self.last_open_dir) / label / subPath
    else:
        dst_path = pathlib.Path(self.last_open_dir) / flagStr / label / subPath
    dst_path.mkdir(parents=True, exist_ok=True)
    # create output filename
    #dst_file = dst_path / f"{orig_filename}_{format(score, '.2f')}.jpg"
    dst_file = dst_path / f"{orig_filename}.jpg"
    # Save image safely handling non-ASCII paths
    is_success, buf = cv2.imencode(".jpg", crop_image)
    if is_success:
        buf.tofile(str(dst_file))

# 新写的批量自动标定
#   新增了: 移除归并类的逻辑（需要在*.yaml文件中, 给模型中给归并类的描述添加"(gbl)"后缀）
#          说明：模型中可能会有归并类（动物 = 猫 + 狗)， 人工标定时无需关心归并类， 故不保存归并类
#   新增了: 将权重小于0.9的子图, 按权重分文件夹保存(用于发现标定错的)；
#          说明：权重大于0.9的占比很大，且99.9%都是正确，无需人工查看， 故不用保存子图
#   新增了: 将所有未检到目标的原图收集到background(用于发现未标定的)
#          若没有原标定, 只有新检测, 则将新检测是背景的全收集到一起
#          若即有原标定, 又有新检测, 则将 原标定 和 新检测 中仅有一个是背景的收集到一起
def save_auto_result_Extend(self, image_file, auto_labeling_result):
    # 生成标签文件
    save_auto_labeling_result(self, image_file, auto_labeling_result)

    # 提取子图, 保存到指定文件夹中
    for shape in auto_labeling_result.shapes:
        if shape.score < 0.9:
            crop_save_zitu(self, image_file, shape.label, shape.points, shape.score)

    # 提取背景图, 保存到background中
    backFlag = 0
    if auto_labeling_result.replace:
        # 若没有标定, 只有新检测, 则将新检测是背景的全收集到一起
        if len(auto_labeling_result.shapes) == 0:
            backFlag = 1
    else:
        # 将原标定是背景, 新检测却不是背景的图收集到一起
        if len(auto_labeling_result.shapes) > 0 and len(self.canvas.shapes) == 0:
            backFlag = 1
        # 将原标定不是背景, 新检测却是背景的图收集到一起
        if len(auto_labeling_result.shapes) == 0 and len(self.canvas.shapes) > 0:
            backFlag = 1
    if backFlag:
        imgPath = pathlib.Path(image_file)
        dst_path = pathlib.Path(self.last_open_dir) / "background"
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_file = dst_path / f"{imgPath.stem}.jpg"
        shutil.copy(image_file, dst_file)

# 提取疑错标签:
#   功能1: 对比已有标签和新检测标签, 细分为7个类别: 正确的, 多检的, 位置偏, 仅归并, 难区分, 无归并, 少检的
#   功能2: 将疑似错误的标签和原图提取出来, 存到指定文件夹中(功能1定义的7个类别名称)
#   功能3: 保存子图时, 先按疑似错误类别分一级文件夹, 然后再按权重分二级文件夹, 更有利于人工挑错
#   功能4: 保存子图时, 也保存背景图(子图用于发现标错的, 背景图用于发现没标的)
def save_auto_result_FindErr(self, image_file, auto_labeling_result):
    def calculate_iou(box1, box2):
        # Calculate the intersection area
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        # Calculate the union region
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        # Calculate IOU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    # 读取标签文件中的标签坐标
    label_file = osp.splitext(image_file)[0] + ".json"
    with io_open(label_file, "r") as f:
        data = json.load(f)
    yuan_shapes = data["shapes"]

    flagStr = ["正确的", "多检的", "位置偏", "仅归并", "难区分", "无归并", "少检的"]
    count = [0,0,0,0,0,0,0]
    badResultShapes = []
    guibingShapes = []
    pos = 0
    for shape1 in auto_labeling_result.shapes:
        shape1.errorTypeStr = flagStr[0]
        pos = pos + 1
        if shape1.label.find("(gbl)") != -1: # 收集归并类
            guibingShapes.append(shape1)
            guibingShapes[-1].my_index = pos - 1
        else:
            points = shape1.points
            a = [points[0].x(), points[0].y(), points[2].x(), points[2].y()]
            flag = 1  # 多检的
            for shape2 in yuan_shapes:
                points = shape2["points"]
                b = [points[0][0], points[0][1], points[2][0], points[2][1]]
                iou = calculate_iou(a, b)
                if iou > 0.85:
                    # 仅在标签相同时比较iou，否则将其视为[多检的/少检的]
                    if shape1.label == shape2["label"]:
                        flag = 0  # 正确的
                    else:
                        flag = 4  # 难区分
                elif iou > 0.35:
                    flag = 2  # 位置偏
            shape1.errorTypeStr = flagStr[flag]
            count[flag] += 1 # 0-正确的，1-多检的，2-位置偏，4-难区分
            if flag != 0: #收集疑似错误的标签
                badResultShapes.append(shape1)

    # 归并类的处理1: 0 0 5 NO;   3 0 5 OK;   0 4 5 OK;   3 4 5 NO
    for shape2 in guibingShapes:
        points = shape2.points
        a = [points[0].x(), points[0].y(), points[2].x(), points[2].y()]
        flag = 0
        for shape1 in auto_labeling_result.shapes:
            if shape1.label.find("(gbl)") != -1:
                continue
            points = shape1.points
            b = [points[0].x(), points[0].y(), points[2].x(), points[2].y()]
            iou = calculate_iou(a, b)
            if iou > 0.85:
                flag += 1
                if shape1.label not in self.guibinglabels:
                    self.guibinglabels.append(shape1.label)
        if flag == 0:
            count[3] += 1   #仅归并
            auto_labeling_result.shapes[shape2.my_index].errorTypeStr = flagStr[3]
            badResultShapes.append(shape2)
        if flag >= 2:
            count[4] += 1   #难区分
            auto_labeling_result.shapes[shape2.my_index].errorTypeStr = flagStr[4]

    #归并类的处理2: 3 0 0 NO;  0 4 0 NO;   3 4 0 NO
    for shape1 in auto_labeling_result.shapes:
        if shape1.label.find("(gbl)") != -1:
            continue
        points = shape1.points
        a = [points[0].x(), points[0].y(), points[2].x(), points[2].y()]
        flag = 0
        for shape2 in guibingShapes:
            points = shape2.points
            b = [points[0].x(), points[0].y(), points[2].x(), points[2].y()]
            iou = calculate_iou(a, b)
            if iou > 0.85:
                flag += 1
        if flag == 0 and (shape1.label in self.guibinglabels):
            count[5] += 1   #无归并

    countA = len(auto_labeling_result.shapes) - len(guibingShapes)
    countB = len(self.canvas.shapes)
    if countA < countB and countB > 0:
        count[6] += 1       #少检的

    # 追加疑似错误标签到标签列表中
    #self.canvas.load_shapes(badResultShapes, replace=False)

    # Only transfer tags that may have issues
    for i in range(1, 6, 1):
        if count[i] > 0 and (countA != countB or countA != count[0] or count[3] > 0):
            # 保存图片文件
            dirname, filename = os.path.split(image_file)
            pic_file = pathlib.Path(self.last_open_dir).joinpath(flagStr[i])
            pic_file.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_file, pic_file / filename)
            # 保存标签文件
            label_file = osp.splitext(image_file)[0] + ".json"
            dirname, filename = os.path.split(label_file)
            label_file = pathlib.Path(self.last_open_dir).joinpath(flagStr[i], filename)
            new_shapes = [shape.to_dict() for shape in badResultShapes]
            data["shapes"].extend(new_shapes)
            with io_open(label_file, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # Crop and save image
    for shape in auto_labeling_result.shapes:
        if shape.score < 0.9 or shape.errorTypeStr != flagStr[0]:
            crop_save_zitu(self, image_file, shape.label, shape.points, shape.score, shape.errorTypeStr)

    # 若即有原标定, 又有新检测, 则将 原标定 和 新检测 中仅有一个是背景的收集到一起
    if ((len(auto_labeling_result.shapes) > 0 and len(self.canvas.shapes) == 0)
            or (len(auto_labeling_result.shapes) == 0 and len(self.canvas.shapes) > 0)):
        imgPath = pathlib.Path(image_file)
        dst_path = pathlib.Path(self.last_open_dir) / "background"
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_file = dst_path / f"{imgPath.stem}.jpg"
        shutil.copy(image_file, dst_file)

# 剔除极值样本(极大值, 极小值), 并保存到指定文件夹中:
def save_auto_result_MinMax(self, image_file, auto_labeling_result):
    """Apply auto labeling results to the current image."""
    if not self.image or not self.image_path:
        return

    # todo


# 极小样本擦除:
#   功能1: 根据规则, 直接擦除;
#   功能2: 每发现一个弹框询问一次, 人为决定是否擦除
def save_auto_result_Erase(self, image_file, auto_labeling_result):
    """Apply auto labeling results to the current image."""
    if not self.image or not self.image_path:
        return

    # todo

def process_next_image(self, progress_dialog):
    try:
        total_images = len(self.image_tuple) #list太慢了, 1万张耗时约10ms, 为了加速换成了tuple

        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            t1_start = time.time()
        
            image_file = self.image_tuple[self.image_index] #list太慢了, 1万张耗时约10ms, 为了加速换成了tuple
            
            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=True,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=True,
                    )
                )
            else:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image, image_file, batch=True
                    )
                )

            if self.dealFunFlag == 0:
                save_auto_labeling_result(self, image_file, auto_labeling_result)
            elif self.dealFunFlag == 1:
                save_auto_result_Extend(self, image_file, auto_labeling_result)
            elif self.dealFunFlag == 2:
                save_auto_result_FindErr(self, image_file, auto_labeling_result)
            elif self.dealFunFlag == 3:
                save_auto_result_MinMax(self, image_file, auto_labeling_result)
            else:
                save_auto_result_Erase(self, image_file, auto_labeling_result)

            progress_dialog.setValue(self.image_index)
            self.image_index += 1

            # 进度和耗时打印
            if self.image_index < 10 or self.image_index % 16 == 0:
                t1 = 1000 * (time.time() - t1_start)
                tall = (time.time() - self.start_time_all)
                tshy = tall * (total_images - self.image_index) / self.image_index
                print(f"进度{self.image_index}/{total_images}, 已耗时{tall:.2f}s, 还剩余{tshy:.2f}s, 耗时{t1:.2f}ms, {self.filename}")

        t1 = 1000 * (time.time() - t1_start)
        tall = (time.time() - self.start_time_all)
        tshy = tall * (total_images - self.image_index) / self.image_index
        print(f"进度{self.image_index}/{total_images}, 已耗时{tall:.2f}s, 还剩余{tshy:.2f}s, 耗时{t1:.2f}ms, {self.filename}")

        finish_processing(self, progress_dialog)

    except Exception as e:
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def show_progress_dialog_and_process(self):
    self.cancel_processing = False

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        self.image_index,
        len(self.image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)

    progress_dialog.setLabelText(
        f"Progress: {self.image_index}/{len(self.image_list)}"
    )
    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:

        def update_progress(value):
            progress_dialog.setLabelText(f"{value}/{len(self.image_list)}")

        progress_bar.valueChanged.connect(update_progress)

    progress_dialog.setStyleSheet(
        """
        QProgressDialog {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08),
                        0 2px 6px rgba(0, 0, 0, 0.04);
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: rgba(0, 0, 0, 0.05);
            text-align: center;
            color: #1d1d1f;
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0066FF,
                stop:0.5 #00A6FF,
                stop:1 #0066FF);
            border-radius: 3px;
        }
        QLabel {
            color: #1d1d1f;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        QPushButton {
            background-color: rgba(255, 255, 255, 0.8);
            border: 0.5px solid rgba(0, 0, 0, 0.1);
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: #0066FF;
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        QPushButton:pressed {
            background-color: rgba(0, 0, 0, 0.08);
        }
    """
    )
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
    if len(self.image_list) < 1:
        return
        
    self.image_tuple = tuple(self.image_list)
    self.start_time_all = time.time() 

    if self.auto_labeling_widget.model_manager.loaded_model_config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in INVALID_MODEL_LIST
    ):
        logger.warning(
            f"The model `{self.auto_labeling_widget.model_manager.loaded_model_config['type']}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Confirmation"))
    response.setText(self.tr("Do you want to process all images?"))
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    logger.info("Start running all images...")

    self.current_index = self.fn_to_index[str(self.filename)]
    self.image_index = self.current_index
    self.text_prompt = ""
    self.run_tracker = False

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in TEXT_PROMPT_MODELS
    ):
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt:
            show_progress_dialog_and_process(self)

    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in VIDEO_MODELS
    ):
        self.run_tracker = True
        show_progress_dialog_and_process(self)

    else:
        show_progress_dialog_and_process(self)
