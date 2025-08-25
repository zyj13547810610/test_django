from django.http import JsonResponse
from ..services import DashboardService
from rest_framework.decorators import api_view
from drf_yasg.utils import swagger_auto_schema

from rest_framework.response import Response
from rest_framework.exceptions import APIException, ValidationError
from backend_re.backend.models import AlertType
from rest_framework import serializers

@api_view(["GET"])
def get_system_status(request):
    """获取系统状态信息"""
    service = DashboardService()
    data = service.get_system_status()
    return JsonResponse(data)

@api_view(["GET"])
def get_detection_stats(request):
    """获取今日检测统计"""
    service = DashboardService()
    data = service.get_detection_stats()
    return JsonResponse(data)

@api_view(["GET"])
def get_defect_stats(request):
    """获取缺陷统计"""
    service = DashboardService()
    data = service.get_defect_stats()
    return JsonResponse(data)

@api_view(["GET"])
def get_accuracy(request):
    """获取检测准确率统计"""
    service = DashboardService()
    data = service.get_accuracy()
    return JsonResponse(data)

@api_view(["GET"])
def get_recent_alerts(request):
    """获取最近告警"""
    limit = int(request.query_params.get("limit", 10))
    service = DashboardService()
    data = service.get_recent_alerts(limit=limit)
    return JsonResponse(data,safe=False)




class CreateAlertSerializer(serializers.Serializer):
    message = serializers.CharField()
    alert_type = serializers.ChoiceField(choices=["INFO", "WARNING", "ERROR"], default="INFO")
    device_id = serializers.IntegerField(required=False)

@swagger_auto_schema(method='post', request_body=CreateAlertSerializer)
@api_view(["POST"])
def create_alert(request):
    serializer = CreateAlertSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    alert_type_enum = AlertType[serializer.validated_data["alert_type"]]
    service = DashboardService()
    alert = service.create_alert(**serializer.validated_data)

    if not alert:
        raise APIException(detail="创建告警失败", code=500)

    return Response({"id": str(alert.id), "message": "告警创建成功"})


# @api_view(["POST"])
# def create_alert(request):
#     """创建新的告警"""
#     message = request.data.get("message")
#     alert_type = request.data.get("alert_type", "INFO")
#     device_id = request.data.get("device_id")

#     try:
#         alert_type_enum = AlertType[alert_type]
#     except KeyError:
#         alert_type_enum = AlertType.INFO

#     service = DashboardService()
#     alert = service.create_alert(
#         message=message,
#         alert_type=alert_type_enum,
#         device_id=device_id
#     )

#     if not alert:
#         raise APIException(detail="创建告警失败", code=500)

#     return JsonResponse({"id": str(alert.id), "message": "告警创建成功"})


@api_view(["PUT"])
def mark_alert_as_read(request, alert_id):
    """将告警标记为已读"""
    service = DashboardService()
    success = service.mark_alert_as_read(alert_id)

    if not success:
        raise APIException(detail="告警不存在或更新失败", code=404)

    return Response({"message": "已将告警标记为已读"})


@api_view(["GET"])
def get_device_status_summary(request):
    """获取设备状态摘要"""
    service = DashboardService()
    data = service.get_device_status_summary()
    return JsonResponse(data)


@api_view(["GET"])
def get_daily_detection_trends(request):
    """获取每日检测趋势"""
    days = int(request.query_params.get("days", 7))
    if not (1 <= days <= 30):
        raise ValidationError(detail="Days must be between 1 and 30")
    service = DashboardService()
    data = service.get_daily_detection_trends(days=days)
    return Response(data)


