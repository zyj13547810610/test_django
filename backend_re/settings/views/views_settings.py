from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from typing import Dict, Any, Optional
from ..services import SettingsService
from backend_re.backend.models import Settings, Device
from rest_framework import serializers
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils.translation import gettext_lazy as _
# --- Serializers ---
class SettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Settings
        fields = ["category", "key", "value", "description"]

class SystemSettingsUpdateSerializer(serializers.Serializer):
    auto_save_interval = serializers.IntegerField(required=False)
    data_retention = serializers.CharField(required=False)
    alarm_threshold = serializers.IntegerField(required=False)
    language = serializers.CharField(required=False)

class MesSettingsUpdateSerializer(serializers.Serializer):
    server_url = serializers.CharField(required=False)
    api_key = serializers.CharField(required=False)

class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = ["id", "name", "type", "model", "status", "config"]

class DeviceCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    type = serializers.CharField()
    model = serializers.CharField(required=False, allow_null=True)
    config = serializers.DictField(child=serializers.CharField(), required=False)

class DeviceUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    type = serializers.CharField(required=False)
    model = serializers.CharField(required=False, allow_null=True)
    status = serializers.CharField(required=False)
    config = serializers.DictField(child=serializers.CharField(), required=False)

# --- ViewSet ---
class SettingsViewSet(viewsets.ViewSet):
    """
    Settings & Devices API
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.service = SettingsService()

    # --- General Settings ---
    @swagger_auto_schema(
        operation_description="获取指定类别或所有设置",
        # operation_summary="获取指定类别或所有设置",
        # operation_id="get_all",
        manual_parameters=[
            openapi.Parameter(
                "category",
                openapi.IN_QUERY,
                description="设置类别",
                type=openapi.TYPE_STRING,
                required=False
            )
        ],
        responses={
            200: SettingSerializer(many=True),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=False, methods=["get"])
    def get_all(self, request):
        '''获取指定类别或所有设置'''
        category = request.query_params.get("category")
        settings = self.service.get_settings(category)
        serializer = SettingSerializer(settings, many=True)
        return Response(serializer.data)

    # --- System Settings ---
    
    @swagger_auto_schema(
        operation_description="获取所有系统设置",
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "auto_save_interval": openapi.Schema(type=openapi.TYPE_INTEGER),
                    "data_retention": openapi.Schema(type=openapi.TYPE_STRING),
                    "alarm_threshold": openapi.Schema(type=openapi.TYPE_INTEGER),
                    "language": openapi.Schema(type=openapi.TYPE_STRING),
                }
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=False, methods=["get"])
    def get_system(self, request):
        '''获取所有系统设置'''
        settings = self.service.get_system_settings()
        return Response(settings)
    
    
    @swagger_auto_schema(
        operation_description="更新系统设置",
        request_body=SystemSettingsUpdateSerializer,
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "auto_save_interval": openapi.Schema(type=openapi.TYPE_INTEGER),
                    "data_retention": openapi.Schema(type=openapi.TYPE_STRING),
                    "alarm_threshold": openapi.Schema(type=openapi.TYPE_INTEGER),
                    "language": openapi.Schema(type=openapi.TYPE_STRING),
                }
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @get_system.mapping.put
    def update_system(self, request):
        '''更新系统设置'''
        serializer = SystemSettingsUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        settings_dict = {k: v for k, v in serializer.validated_data.items() if v is not None}
        updated = self.service.update_system_settings(settings_dict)
        return Response(updated)

    # --- MES Settings ---
    
    @swagger_auto_schema(
        operation_description="获取MES集成设置",
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                additional_properties=openapi.Schema(type=openapi.TYPE_STRING)
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=False, methods=["get"])
    def get_mes(self, request):
        '''获取MES集成设置'''
        settings = self.service.get_mes_settings()
        return Response(settings)


    @swagger_auto_schema(
        operation_description="更新MES集成设置",
        request_body=MesSettingsUpdateSerializer,
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                additional_properties=openapi.Schema(type=openapi.TYPE_STRING)
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @get_mes.mapping.put
    def update_mes(self, request):
        '''更新MES集成设置'''
        serializer = MesSettingsUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        settings_dict = {k: v for k, v in serializer.validated_data.items() if v is not None}
        updated = self.service.update_mes_settings(settings_dict)
        return Response(updated)

    # --- Devices ---
    @swagger_auto_schema(
        operation_description="获取设备列表",
        manual_parameters=[
            openapi.Parameter(
                "type",
                openapi.IN_QUERY,
                description="设备类型",
                type=openapi.TYPE_STRING,
                required=False
            )
        ],
        responses={
            200: DeviceSerializer(many=True),
            500: "获取设备列表失败"
        }
    )
    @action(detail=False, methods=["get"])
    def get_devices(self, request):
        '''获取设备列表'''
        type_filter = request.query_params.get("type")
        devices = self.service.get_devices(type_filter)
        serializer = DeviceSerializer(devices, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        operation_description="获取单个设备（pk 为设备ID）",
        # manual_parameters=[
        #     openapi.Parameter(
        #         "device_id",
        #         openapi.IN_PATH,
        #         description="设备ID",
        #         type=openapi.TYPE_INTEGER,
        #         required=True
        #     )
        # ],
        responses={
            200: DeviceSerializer,
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=True, methods=["get"])
    def get_device(self, request, pk=None):
        '''获取单个设备'''
        device = self.service.get_device(pk)
        if not device:
            return Response({"detail": "Device not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = DeviceSerializer(device)
        return Response(serializer.data)


    @swagger_auto_schema(
        operation_description="创建新设备",
        request_body=DeviceCreateSerializer,
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                additional_properties=openapi.Schema(type=openapi.TYPE_STRING)
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )

    @action(detail=False, methods=["post"])
    def create_device(self, request):
        '''创建新设备'''
        serializer = DeviceCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        device = self.service.create_device(
            name=serializer.validated_data["name"],
            type=serializer.validated_data["type"],
            model=serializer.validated_data.get("model"),
            config=serializer.validated_data.get("config", {})
        )
        return Response(DeviceSerializer(device).data)

    @swagger_auto_schema(
        operation_description="更新设备(（pk 为设备ID）)",
        # manual_parameters=[
        #     openapi.Parameter(
        #         "device_id",
        #         openapi.IN_PATH,
        #         description="设备ID",
        #         type=openapi.TYPE_INTEGER,
        #         required=True
        #     )
        # ],
        request_body=DeviceUpdateSerializer,
        responses={
            200: DeviceSerializer,
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=True, methods=["put"])
    def update_device(self, request, pk=None):
        '''更新设备'''
        serializer = DeviceUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        device_dict = {k: v for k, v in serializer.validated_data.items() if v is not None}
        device = self.service.update_device(pk, device_dict)
        return Response(DeviceSerializer(device).data)
    
    @swagger_auto_schema(
        operation_description="删除设备(pk 为设备ID）)",
        # manual_parameters=[
        #     openapi.Parameter(
        #         "device_id",
        #         openapi.IN_PATH,
        #         description="设备ID",
        #         type=openapi.TYPE_INTEGER,
        #         required=True
        #     )
        # ],
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"message": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=True, methods=["delete"])
    def delete_device(self, request, pk=None):
        '''删除设备'''
        success = self.service.delete_device(pk)
        if not success:
            return Response({"detail": "Device not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"message": "Device deleted successfully"})


    @swagger_auto_schema(
        operation_description="更新设备状态",
        manual_parameters=[
            # openapi.Parameter(
            #     "device_id",
            #     openapi.IN_PATH,
            #     description="设备ID",
            #     type=openapi.TYPE_INTEGER,
            #     required=True
            # ),
            openapi.Parameter(
                "status",
                openapi.IN_QUERY,
                description="设备状态 (online, offline, error)",
                type=openapi.TYPE_STRING,
                required=True,
                enum=["online", "offline", "error"]
            )
        ],
        responses={
            200: DeviceSerializer,
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={"detail": openapi.Schema(type=openapi.TYPE_STRING)}
            )
        }
    )
    @action(detail=True, methods=["put"])
    def update_device_status(self, request, pk=None):
        '''更新设备状态'''
        status_value = request.query_params.get("status")
        if not status_value:
            return Response({"detail": "Missing status"}, status=status.HTTP_400_BAD_REQUEST)
        device = self.service.update_device_status(pk, status_value)
        return Response(DeviceSerializer(device).data)
