from rest_framework import serializers
import json
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.shortcuts import get_object_or_404

from ..services import CVOperationService
from backend_re.backend.models import CVOperation
# from backend_re.modles_train.services import ModelService    # YOLO model service

# -------------------------------
# 参数配置序列化器
# -------------------------------
class ParamConfigSerializer(serializers.Serializer):
    name = serializers.CharField(required=True)
    type = serializers.ChoiceField(choices=["text", "number", "boolean", "array", "object", "image"])
    description = serializers.CharField(required=False, allow_null=True)
    default = serializers.JSONField(required=False, allow_null=True)
    required = serializers.BooleanField(default=False)

    def validate_name(self, value):
        if not value.isidentifier():
            raise serializers.ValidationError("参数名称必须是有效的Python标识符")
        return value

    def validate(self, attrs):
        param_type = attrs.get("type")
        default = attrs.get("default")

        if default is None:
            return attrs

        # 类型检查
        if param_type == "text" and isinstance(default, str):
            return attrs
        elif param_type == "number" and isinstance(default, (int, float)):
            return attrs
        elif param_type == "boolean" and isinstance(default, bool):
            return attrs
        elif param_type == "array" and isinstance(default, list):
            return attrs
        elif param_type == "object" and isinstance(default, dict):
            return attrs
        elif param_type == "image" and isinstance(default, str):
            return attrs
        
        # 如果是字符串，尝试转换
        if isinstance(default, str):
            try:
                if param_type == "number":
                    attrs["default"] = float(default) if "." in default else int(default)
                elif param_type == "boolean":
                    attrs["default"] = default.lower() in ("true", "1", "yes", "y")
                elif param_type in ("array", "object"):
                    attrs["default"] = json.loads(default)
                # text 和 image 保持字符串
                return attrs
            except Exception as e:
                raise serializers.ValidationError(f"无法将字符串转换为 {param_type}: {str(e)}")

        raise serializers.ValidationError(f"{param_type} 类型参数的默认值不合法")


# -------------------------------
# CVOperation 响应
# -------------------------------
class CVOperationResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField(min_length=1, max_length=100)
    description = serializers.CharField(required=False, allow_null=True)
    code = serializers.CharField()
    # inputParams = ParamConfigSerializer(many=True, default=list)
    # outputParams = ParamConfigSerializer(many=True, default=list)
    # createdAt = serializers.DateTimeField(help_text="创建时间")
    # updatedAt = serializers.DateTimeField(help_text="更新时间")
    inputParams = ParamConfigSerializer(many=True, source='input_params',default=list)
    outputParams = ParamConfigSerializer(many=True, source='output_params',default=list)
    createdAt = serializers.DateTimeField(source='created_at',help_text="创建时间")
    updatedAt = serializers.DateTimeField(source='updated_at',help_text="更新时间")
    class Meta:
        swagger_schema_fields = {
            "example": {
                "id": 1,
                "name": "图像预处理",
                "description": "对图像进行预处理操作",
                "code": "def process(image):\n    return cv2.resize(image, (224, 224))",
                "inputParams": [
                    {
                        "name": "image",
                        "type": "object",
                        "description": "输入图像",
                        "required": True
                    }
                ],
                "outputParams": [
                    {
                        "name": "result",
                        "type": "object",
                        "description": "处理后的图像",
                        "required": True
                    }
                ],
                "createdAt": "2024-03-22T10:00:00",
                "updatedAt": "2024-03-22T10:00:00"
            }
        }

# -------------------------------
# 应用操作响应
# -------------------------------
class ApplyOperationResponseSerializer(serializers.Serializer):
    result = serializers.DictField()

    def validate_result(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("结果必须是字典类型")
        return value


# -------------------------------
# 请求模型
# -------------------------------
class CreateOperationRequestSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=1, max_length=100)
    description = serializers.CharField(required=False, allow_null=True)
    code = serializers.CharField()
    inputParams = ParamConfigSerializer(many=True, required=False)
    outputParams = ParamConfigSerializer(many=True, required=False)

    def validate_code(self, value):
        if not value.strip():
            raise serializers.ValidationError("代码不能为空")
        return value.strip()


class UpdateOperationRequestSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=1, max_length=100, required=False, allow_null=True)
    description = serializers.CharField(required=False, allow_null=True)
    code = serializers.CharField(required=False, allow_null=True)
    inputParams = ParamConfigSerializer(many=True, required=False)
    outputParams = ParamConfigSerializer(many=True, required=False)

    def validate_code(self, value):
        if value is not None and not value.strip():
            raise serializers.ValidationError("代码不能为空")
        return value.strip() if value is not None else value


class ApplyOperationRequestSerializer(serializers.Serializer):
    inputParams = serializers.DictField()



class CVOperationViewSet(viewsets.ViewSet):
    """
    CV 操作相关接口
    """

    @swagger_auto_schema(
        responses={200: CVOperationResponseSerializer(many=True)},
        operation_description="获取所有CV操作"
    )
    @action(detail=False, methods=["get"],url_path='get_operations')
    def get_operations(self, request):
        service = CVOperationService()
        operations = service.get_operations()
        serializer = CVOperationResponseSerializer(operations, many=True)
#         serializer = CVOperationResponseSerializer([
#     {
#         "id": op.id,
#         "name": op.name,
#         "description": op.description,
#         "code": op.code,
#         "inputParams": op.input_params,
#         "outputParams": op.output_params,
#         "createdAt": op.created_at,
#         "updatedAt": op.updated_at,
#     } for op in operations
# ], many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        responses={200: CVOperationResponseSerializer()},
        operation_description="获取单个CV操作"
    )
    @action(detail=True, methods=["get"],url_path='get_operation')
    def get_operation(self, request, pk=None):
        service = CVOperationService()
        operation = service.get_operation(pk)
        if not operation:
            return Response({"detail": "Operation not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = CVOperationResponseSerializer(operation)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=CreateOperationRequestSerializer,
        responses={201: CVOperationResponseSerializer()},
        operation_description="创建新的CV操作"
    )
    @action(detail=False, methods=["post"],url_path='create_operation')
    def create_operation(self, request):
        serializer = CreateOperationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        service = CVOperationService()
        
        data = serializer.validated_data.copy()
        data['input_params'] = data.pop('inputParams', [])
        data['output_params'] = data.pop('outputParams', [])
        operation = service.create_operation(**data)
        operation = service.create_operation(**serializer.validated_data)
        return Response(CVOperationResponseSerializer(operation).data, status=status.HTTP_201_CREATED)

    @swagger_auto_schema(
        request_body=UpdateOperationRequestSerializer,
        responses={200: CVOperationResponseSerializer()},
        operation_description="更新CV操作"
    )
    @action(detail=True, methods=["put"],url_path='update_operation')
    def update_operation(self, request, pk=None):
        serializer = UpdateOperationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        service = CVOperationService()
        operation = service.update_operation(pk, **serializer.validated_data)
        return Response(CVOperationResponseSerializer(operation).data)

    @swagger_auto_schema(
        operation_description="删除CV操作",
        responses={204: "No content"}
    )
    @action(detail=True, methods=["delete"],url_path='delete_operation')
    def delete_operation(self, request, pk=None):
        service = CVOperationService()
        service.delete_operation(pk)
        return Response(status=status.HTTP_204_NO_CONTENT)

    @swagger_auto_schema(
        method="post",
        request_body=ApplyOperationRequestSerializer,
        responses={200: ApplyOperationResponseSerializer()},
        operation_description="应用CV操作"
    )
    @action(detail=True, methods=["post"])
    def apply_operation(self, request, pk=None):
        serializer = ApplyOperationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        service = CVOperationService()
        try:
            result = service.apply_operation(pk, serializer.validated_data["inputParams"])
            return Response(ApplyOperationResponseSerializer({"result": result}).data)
        except Exception as e:
            return Response({"detail": f"操作执行失败: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)