from rest_framework import serializers
from backend_re.backend.models import DetectionStatus, Detection

# 驼峰命名工具
def to_camel_case(snake_str: str) -> str:
    parts = snake_str.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

class CamelModelSerializer(serializers.ModelSerializer):
    def to_representation(self, instance):
        """输出驼峰风格字段"""
        ret = super().to_representation(instance)
        return {to_camel_case(k): v for k, v in ret.items()}

# class DetectionCreateSerializer(CamelModelSerializer):
#     class Meta:
#         model = Detection
#         fields = [
#             "text", "confidence", "status", "device_id",
#             "image_data", "processed_image_data", "operation_id",
#             "operation_type", "metadata"
#         ]
class DetectionCreateSerializer(serializers.Serializer):
    text = serializers.CharField(required=False, allow_blank=True)
    confidence = serializers.FloatField(required=False)
    status = serializers.CharField(required=False)
    device_id = serializers.IntegerField(required=False)
    image_data = serializers.CharField(required=False, allow_blank=True)
    processed_image_data = serializers.CharField(required=False, allow_blank=True)
    operation_id = serializers.IntegerField(required=False)
    operation_type = serializers.CharField(required=False, allow_blank=True)
    metadata = serializers.JSONField(required=False)
    
# class DetectionResponseSerializer(CamelModelSerializer):
#     device_name = serializers.CharField(source="device.name", allow_null=True)

#     class Meta:
#         model = Detection
#         fields = [
#             "id", "text", "confidence", "status", "device_id",
#             "device_name", "timestamp", "image_path",
#             "processed_image_path", "operation_id",
#             "operation_type", "metadata"
#         ]

class DetectionResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    text = serializers.CharField(required=False, allow_blank=True)
    confidence = serializers.FloatField(required=False)
    status = serializers.CharField()
    device_id = serializers.IntegerField(required=False)
    device_name = serializers.CharField(required=False, allow_null=True)
    timestamp = serializers.DateTimeField()
    image_path = serializers.CharField(required=False, allow_blank=True)
    processed_image_path = serializers.CharField(required=False, allow_blank=True)
    operation_id = serializers.IntegerField(required=False)
    operation_type = serializers.CharField(required=False, allow_blank=True)
    metadata = serializers.JSONField(required=False)  # 可以返回任意 JSON


class PaginatedDetectionResponseSerializer(serializers.Serializer):
    items = DetectionResponseSerializer(many=True)
    total = serializers.IntegerField()
    page = serializers.IntegerField()
    per_page = serializers.IntegerField()
    pages = serializers.IntegerField()
