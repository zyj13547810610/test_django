from django.db import models
from django.utils import timezone
from typing import TypedDict, Any
from typing_extensions import TypedDict
from enum import Enum as PyEnum

class DatasetType(PyEnum):
    TEXT_REGION = 'text_region'  # 文本区域数据集
    OCR = 'ocr'                  # OCR数据集
DATASET_TYPE_CHOICES = [(e.value, e.value) for e in DatasetType]

class Dataset(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=20, choices=DATASET_TYPE_CHOICES)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    external_dataset_url = models.URLField(blank=True, null=True) 
    class Meta:
        db_table = 'datasets'


class Image(models.Model):
    id = models.AutoField(primary_key=True)
    filename=models.CharField(max_length=255)
    url=models.CharField(max_length=1000)
    created_at = models.DateTimeField(default=timezone.now)
    is_annotated = models.BooleanField(default=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='images')
    class Meta:
        db_table = 'images'


class Annotation(models.Model):
    id = models.AutoField(primary_key=True)
    data = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    image=models.ForeignKey(Image, on_delete=models.CASCADE, related_name='annotations')
    class Meta:
        db_table='annotations'


class ParamType(PyEnum):
    IMAGE = 'image'           # 图像类型
    NUMBER = 'number'         # 数值类型(int/float)
    TEXT = 'text'            # 文本类型
    BOOLEAN = 'boolean'      # 布尔类型
    ARRAY = 'array'          # 数组类型
    OBJECT = 'object'        # 对象类型
PARAM_TYPE_CHOICES = [(e.value, e.value) for e in ParamType]

class ParamConfig(TypedDict):
    name: str                # 参数名称
    type: ParamType         # 参数类型
    description: str        # 参数描述
    default: Any           # 默认值
    required: bool         # 是否必需

class NodeType(PyEnum):
    START = 'start'         # 开始节点
    END = 'end'            # 结束节点
    OPERATION = 'operation' # 处理节点
    PARALLEL = 'parallel'  # 并行节点
    MERGE = 'merge'       # 聚合节点
    CONDITION = 'condition' # 条件节点
NODE_TYPE_CHOICES = [(e.value, e.value) for e in NodeType]

class EdgeType(PyEnum):
    NORMAL = 'normal'  # 普通连接
    TRUE = 'true'     # 条件为真的分支
    FALSE = 'false'   # 条件为假的分支
EDGE_TYPE_CHOICES = [(e.value, e.value) for e in EdgeType]

class Pipeline(models.Model):
    """流水线模型"""
    id = models.AutoField(primary_key=True)
    name =models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    pipeline_metadata = models.JSONField(default=dict)
    input_params = models.JSONField(default=list)
    output_params = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    class Meta:
        db_table='pipelines'
    
class CVOperation(models.Model):   
    id = models.AutoField(primary_key=True)
    name =models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    code = models.TextField()
    input_params = models.JSONField(default=list)
    output_params = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    class Meta:
        db_table='cv_operations'
  
class ModelStatus(PyEnum):
    NOT_STARTED = 'not_started'
    TRAINING = 'training'
    COMPLETED = 'completed'
    FAILED = 'failed'
MODEL_STATUS_CHOICES = [(e.value, e.value) for e in ModelStatus]
    
class ModelArchitecture(PyEnum):
   
    YOLO_V8 = 'yolo_v8'
    YOLO_V9 = 'yolo_v9'
    YOLO_V10 = 'yolo_v10'
    YOLO_11 = 'yolo_11'
    YOLO_V5 = 'yolo_v5'
MODEL_ARCHITECTURE_CHOICES = [(e.value, e.value) for e in ModelArchitecture]

class Model(models.Model):
    id = models.AutoField(primary_key=True)
    name =models.CharField(max_length=100)
    # architecture= models.CharField(max_length=20, choices=MODEL_ARCHITECTURE_CHOICES)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)  # 添加外键
    status = models.CharField(max_length=20, choices=MODEL_STATUS_CHOICES, default=ModelStatus.NOT_STARTED.value)
    parameters = models.JSONField(default=dict)
    metrics = models.JSONField(blank=True, null=True)
    file_path = models.CharField(max_length=1000, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)
    # class Meta:
    #     db_table='models'
    _architecture = models.CharField(max_length=20, choices=MODEL_ARCHITECTURE_CHOICES, db_column='architecture')

    @property
    def architecture(self):
        return ModelArchitecture(self._architecture)

    @architecture.setter
    def architecture(self, value):
        self._architecture = value.value if isinstance(value, ModelArchitecture) else value

    class Meta:
        db_table = 'models'


# Add this new class for system settings
class Settings(models.Model):
    id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=50)
    key = models.CharField(max_length=100)
    value = models.CharField(max_length=1000)
    description = models.CharField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'settings'
        # Composite unique constraint
        constraints = [
            models.UniqueConstraint(fields=['category', 'key'], name='uq_settings_category_key')
        ]

# Add this class for device settings
class Device(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    type = models.CharField(max_length=50)
    model = models.CharField(max_length=100, blank=True, null=True)
    config = models.JSONField(default=dict)
    status = models.CharField(max_length=20, default='offline')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'devices'
 
# 检测结果状态枚举
class DetectionStatus(PyEnum):
    PASS = 'pass'           # 合格
    FAIL = 'fail'           # 不合格
    UNKNOWN = 'unknown'     # 未知
DETECTION_STATUS_CHOICES = [(e.value, e.value) for e in DetectionStatus]

# 历史检测记录模型
class Detection(models.Model):
    id = models.AutoField(primary_key=True)
    device = models.ForeignKey(Device, on_delete=models.SET_NULL, null=True, related_name='detections')
    timestamp = models.DateTimeField(default=timezone.now)
    text = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=DETECTION_STATUS_CHOICES, default=DetectionStatus.UNKNOWN.value)
    image_path = models.CharField(max_length=1000, blank=True, null=True)
    processed_image_path = models.CharField(max_length=1000, blank=True, null=True)
    operation_id = models.IntegerField(blank=True, null=True)
    operation_type = models.CharField(max_length=50, blank=True, null=True)
    detection_metadata = models.JSONField(blank=True, null=True)

    class Meta:
        db_table = 'detections'

# 告警类型枚举
class AlertType(PyEnum):
    ERROR = 'error'         # 错误
    WARNING = 'warning'     # 警告
    INFO = 'info'           # 信息
ALERT_TYPE_CHOICES = [(e.value, e.value) for e in AlertType]

# 系统告警模型
class Alert(models.Model):
    id = models.AutoField(primary_key=True)
    type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES, default=AlertType.INFO.value)
    message = models.CharField(max_length=500)
    device = models.ForeignKey(Device, on_delete=models.SET_NULL, null=True, related_name='alerts')
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'alerts'



