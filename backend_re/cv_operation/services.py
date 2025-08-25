import base64
import cv2
import numpy as np
from django.utils import timezone
from typing import List, Optional, Dict, Any
from django.core.exceptions import ObjectDoesNotExist
from rest_framework.exceptions import APIException, NotFound, ValidationError

from ..backend.models import CVOperation   # YOLO model service
from backend_re.modles_train.services import ModelService    # YOLO model service


class CVOperationService:
    def __init__(self):
        self.model_service = ModelService()

        # 初始化代码执行上下文
        self.context = {
            'cv2': cv2,
            'np': np,
            'get_yolo_model': self._get_yolo_model,
            'get_all_models': self._get_all_models,
        }

    def _get_yolo_model(self, model_id: int):
        try:
            return self.model_service.get_model_instance(model_id)
        except Exception as e:
            raise ValidationError(f"获取YOLO模型失败: {str(e)}")

    def _get_all_models(self):
        try:
            return self.model_service._loaded_models
        except Exception as e:
            raise ValidationError(f"获取已加载模型失败: {str(e)}")

    # -------------------- CRUD --------------------
    def get_operations(self) -> List[CVOperation]:
        return CVOperation.objects.all()

    def get_operation(self, operation_id: int) -> CVOperation:
        try:
            return CVOperation.objects.get(id=operation_id)
        except ObjectDoesNotExist:
            raise NotFound("Operation not found")

    def create_operation(self, name: str, code: str,
                         input_params: List[Dict] = None,
                         output_params: List[Dict] = None,
                         description: str = None) -> CVOperation:
        try:
            operation = CVOperation.objects.create(
                name=name,
                description=description,
                code=code,
                input_params=input_params or [],
                output_params=output_params or [],
                created_at=timezone.now(),
                updated_at=timezone.now()
            )
            return operation
        except Exception as e:
            raise APIException(f"创建操作失败: {str(e)}")

    def update_operation(self, operation_id: int, **kwargs) -> CVOperation:
        operation = self.get_operation(operation_id)

        for field in ["name", "code", "description", "input_params", "output_params"]:
            if field in kwargs and kwargs[field] is not None:
                setattr(operation, field, kwargs[field])

        operation.updated_at = timezone.now()
        try:
            operation.save()
            return operation
        except Exception as e:
            raise APIException(f"更新操作失败: {str(e)}")

    def delete_operation(self, operation_id: int) -> None:
        operation = self.get_operation(operation_id)
        try:
            operation.delete()
        except Exception as e:
            raise APIException(f"删除操作失败: {str(e)}")

    # -------------------- 执行 --------------------
    def apply_operation(self, operation_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = self.get_operation(operation_id)

        try:
            # 补全参数
            complete_params = {}
            for param_config in operation.input_params:
                param_name = param_config['name']
                param_type = param_config['type']
                param_default = param_config.get('default')

                param_value = params.get(param_name, param_default)

                if param_type == 'image':
                    if isinstance(param_value, str):
                        try:
                            image_data = base64.b64decode(param_value)
                            nparr = np.frombuffer(image_data, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is None:
                                raise ValidationError(f"无效的图像数据: {param_name}")
                            complete_params[param_name] = img
                        except Exception as e:
                            raise ValidationError(f"无法解码图像参数 {param_name}: {str(e)}")
                    elif isinstance(param_value, np.ndarray):
                        complete_params[param_name] = param_value
                    else:
                        complete_params[param_name] = None

                elif param_type == 'number':
                    try:
                        complete_params[param_name] = float(param_value) if param_value is not None else None
                    except (ValueError, TypeError):
                        raise ValidationError(f"无法将参数 {param_name} 转换为数字类型")

                elif param_type == 'boolean':
                    complete_params[param_name] = bool(param_value) if param_value is not None else None

                elif param_type == 'text':
                    complete_params[param_name] = str(param_value) if param_value is not None else None

                elif param_type == 'array':
                    if param_value is None or isinstance(param_value, list):
                        complete_params[param_name] = param_value
                    else:
                        raise ValidationError(f"参数 {param_name} 类型不正确，期望数组")

                elif param_type == 'object':
                    if param_value is None or isinstance(param_value, dict):
                        complete_params[param_name] = param_value
                    else:
                        raise ValidationError(f"参数 {param_name} 类型不正确，期望对象")

                else:
                    complete_params[param_name] = param_value

            # 执行代码
            exec_context = {**self.context, **complete_params}
            namespace = {}
            try:
                exec(operation.code, exec_context, namespace)
            except Exception as e:
                raise ValidationError(f"执行操作代码出错: {str(e)}")

            if 'process' not in namespace or not callable(namespace['process']):
                raise ValidationError("'process' 必须是可调用函数")

            try:
                result = namespace['process'](**complete_params)
            except Exception as e:
                raise ValidationError(f"执行 process 函数出错: {str(e)}")

            if not isinstance(result, dict):
                raise ValidationError(f"process 必须返回字典类型，但返回了 {type(result)}")

            output_data = {}
            for param_config in operation.output_params:
                param_name = param_config['name']
                if param_name not in result:
                    raise ValidationError(f"缺少输出参数: {param_name}")
                output_data[param_name] = result[param_name]

            return output_data

        except APIException:
            raise
        except Exception as e:
            raise APIException(f"服务器内部错误: {str(e)}")
