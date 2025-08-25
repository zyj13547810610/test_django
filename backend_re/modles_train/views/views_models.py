import os
import base64
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404

from backend_re.backend.models import ModelArchitecture
from ..services import ModelService
from rest_framework import serializers, views, status
from rest_framework.response import Response
from django.utils import timezone
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image as PILImage
from django.http import FileResponse, StreamingHttpResponse, JsonResponse
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi



TRAINING_LOGS = {} 
logger = logging.getLogger(__name__)

class ModelCreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    architecture = serializers.ChoiceField(choices=[arch.value for arch in ModelArchitecture])
    dataset_id = serializers.IntegerField(required=False, allow_null=True)
    external_dataset_url = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    parameters = serializers.JSONField(required=False)

class ModelUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(required=False)
    parameters = serializers.JSONField(required=False)

class ModelResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    architecture = serializers.CharField()
    dataset_id = serializers.IntegerField()
    status = serializers.CharField()
    parameters = serializers.JSONField()
    metrics = serializers.JSONField(required=False, allow_null=True)
    created_at = serializers.DateTimeField()
    
    updated_at = serializers.DateTimeField()
    file_path = serializers.CharField(required=False, allow_null=True)

class FileInfoSerializer(serializers.Serializer):
    name = serializers.CharField()
    path = serializers.CharField()
    isDir = serializers.BooleanField()
    size = serializers.IntegerField()
    modifiedTime = serializers.DateTimeField()

class FileListResponseSerializer(serializers.Serializer):
    files = FileInfoSerializer(many=True)

class FileContentResponseSerializer(serializers.Serializer):
    content = serializers.CharField()

class TestModelRequestSerializer(serializers.Serializer):
    image_base64 = serializers.CharField()
    conf_thres = serializers.FloatField(default=0.25)
    iou_thres = serializers.FloatField(default=0.45)
    

class ModelViewSet(viewsets.ViewSet):
    """
    DRF ViewSet for model management
    """
    @action(detail=False, methods=['get'], url_path='get_models')
    def get_models(self, request):
        """所有模型信息 """
        service = ModelService()
        models = service.get_models()
        serializer = ModelResponseSerializer(models, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='get_model')
    def get_model(self, request, pk=None):
        """单个模型信息"""
        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = ModelResponseSerializer(model)
        return Response(serializer.data)


    @swagger_auto_schema(
        request_body=ModelCreateSerializer,
        responses={201: ModelResponseSerializer}
    )
    @action(detail=False, methods=['post'], url_path='create_model')
    def create_model(self, request):
        """新建模型"""
        serializer = ModelCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        service = ModelService()
        model = service.create_model(
            name=data['name'],
            architecture=data['architecture'],
            dataset_id=data.get('dataset_id'),
            external_dataset_url=data.get('external_dataset_url'),
            parameters=data.get('parameters')
        )
        response_serializer = ModelResponseSerializer(model)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


    @swagger_auto_schema(
        request_body=ModelUpdateSerializer,
        responses={201: ModelResponseSerializer}
    )
    # @action(detail=True, methods=['put', 'patch'])
    @action(detail=True, methods=['put'])
    def update_model(self, request, pk=None):
        """更新模型"""
        serializer = ModelUpdateSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        service = ModelService()
        model = service.update_model(pk, **data)
        response_serializer = ModelResponseSerializer(model)
        return Response(response_serializer.data)

    @action(detail=True, methods=['delete'])
    def delete_model(self, request, pk=None):
        """删除模型"""
        service = ModelService()
        service.delete_model(pk)
        return Response({"message": "Model deleted successfully"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def start_training(self, request, pk=None):
        """训练模型"""
        service = ModelService()
        model = service.start_training(pk)
        serializer = ModelResponseSerializer(model)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def export_model(self, request, pk=None):
        """导出模型"""
        service = ModelService()
        export_path = service.export_model(pk)
        return Response({"message": "Model exported successfully", "file_path": export_path})
    
    @swagger_auto_schema(
        request_body=TestModelRequestSerializer,
        responses={201: ModelResponseSerializer}
    )
    @action(detail=True, methods=['post'])
    def test_model_base64(self, request, pk=None):
        """
        使用训练好的模型测试 base64 图像
        """
        serializer = TestModelRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        service = ModelService()
        try:
            image_data = data['image_base64'].split("base64,")[-1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return Response({"success": False, "error": "无法解码图像数据"}, status=400)

            annotated_image, detections = service.test_model_image(
                model_id=pk,
                image=img,
                conf_thres=data['conf_thres'],
                iou_thres=data['iou_thres']
            )

            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return Response({
                "success": True,
                "detections": detections,
                "image": f"data:image/jpeg;base64,{image_base64}"
            })
        except Exception as e:
            return Response({"success": False, "error": str(e)}, status=500)

    @action(detail=True, methods=['get'])
    def get_model_logs(self, request, pk=None):
        """
        获取模型训练日志
        """
        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=404)

        logs = TRAINING_LOGS.get(pk, [])
        if not logs:
            timestamp = time.strftime("%H:%M:%S")
            logs.append(f"[{timestamp}] 等待训练日志...")

        return Response({"model_id": pk, "logs": logs})

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter(
        'path',  # 参数名
        openapi.IN_QUERY,  # 查询参数
        description='文件路径（相对于模型目录）',
        type=openapi.TYPE_STRING,
        required=False
    )], 
        responses={200: FileListResponseSerializer}
    )
    @action(detail=True, methods=['get'])
    def get_model_files(self, request, pk=None):
        """
        获取模型文件列表
        """
        path = request.query_params.get("path", "")
        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=404)

        project_dir = "models"
        model_dir = os.path.abspath(os.path.join(project_dir, f"model_{pk}"))
        target_path = os.path.join(model_dir, path)

        if not os.path.abspath(target_path).startswith(os.path.abspath(model_dir)):
            return Response({"detail": "Access denied"}, status=403)
        if not os.path.exists(target_path):
            return Response({"detail": f"Path not found: {path}"}, status=404)
        if os.path.isdir(target_path):
            files = []
            for item in os.listdir(target_path):
                item_path = os.path.join(target_path, item)
                rel_path = os.path.relpath(item_path, model_dir)
                stat = os.stat(item_path)
                files.append({
                    "name": item,
                    "path": rel_path,
                    "isDir": os.path.isdir(item_path),
                    "size": stat.st_size,
                    "modifiedTime": timezone.datetime.fromtimestamp(stat.st_mtime)
                })
            serializer = FileListResponseSerializer({"files": files})
            return Response(serializer.data)
        else:
            return Response({"detail": "Path is not a directory"}, status=400)


    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter(
        'path',  # 参数名
        openapi.IN_QUERY,  # 查询参数
        description='文件路径（相对于模型目录）',
        type=openapi.TYPE_STRING,
        required=False
    )], 
        responses={200: FileListResponseSerializer}
    )
    @action(detail=True, methods=['get'])
    def get_file_content(self, request, pk=None):
        """
        获取文本文件内容
        """
        path = request.query_params.get("path")
        if not path:
            return Response({"detail": "Missing 'path' parameter"}, status=400)

        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=404)

        project_dir = "models"
        model_dir = os.path.abspath(os.path.join(project_dir, f"model_{pk}"))
        target_path = os.path.join(model_dir, path)

        if not os.path.abspath(target_path).startswith(os.path.abspath(model_dir)):
            return Response({"detail": "Access denied"}, status=403)
        if not os.path.exists(target_path):
            return Response({"detail": f"File not found: {path}"}, status=404)
        if os.path.isdir(target_path):
            return Response({"detail": "Path is a directory, not a file"}, status=400)

        text_extensions = ['.txt', '.csv', '.yaml', '.yml', '.json', '.log', '.md']
        if not any(target_path.lower().endswith(ext) for ext in text_extensions):
            return Response({"detail": "File type not supported for text reading"}, status=400)

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            serializer = FileContentResponseSerializer({"content": content})
            return Response(serializer.data)
        except Exception as e:
            return Response({"detail": f"Failed to read file: {str(e)}"}, status=500)


    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter(
        'path',  # 参数名
        openapi.IN_QUERY,  # 查询参数
        description='文件路径（相对于模型目录）',
        type=openapi.TYPE_STRING,
        required=False
    )], 
        responses={200: FileListResponseSerializer}
    )
    @action(detail=True, methods=['get'])
    def preview_file(self, request, pk=None):
        """
        预览图像文件
        """
        path = request.query_params.get("path")
        if not path:
            return Response({"detail": "Missing 'path' parameter"}, status=400)

        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=404)

        project_dir = "models"
        model_dir = os.path.abspath(os.path.join(project_dir, f"model_{pk}"))
        target_path = os.path.join(model_dir, path)

        if not os.path.abspath(target_path).startswith(os.path.abspath(model_dir)):
            return Response({"detail": "Access denied"}, status=403)
        if not os.path.exists(target_path):
            return Response({"detail": f"File not found: {path}"}, status=404)
        if os.path.isdir(target_path):
            return Response({"detail": "Path is a directory, not a file"}, status=400)

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        if not any(target_path.lower().endswith(ext) for ext in image_extensions):
            return Response({"detail": "File type not supported for preview"}, status=400)

        return FileResponse(target_path, content_type=f"image/{os.path.splitext(target_path)[1][1:]}", filename=os.path.basename(target_path))


    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter(
        'path',  # 参数名
        openapi.IN_QUERY,  # 查询参数
        description='文件路径（相对于模型目录）',
        type=openapi.TYPE_STRING,
        required=False
    )], 
        responses={200: FileListResponseSerializer}
    )
    @action(detail=True, methods=['get'])
    def download_file(self, request, pk=None):
        """
        下载文件
        """
        path = request.query_params.get("path")
        if not path:
            return Response({"detail": "Missing 'path' parameter"}, status=400)

        service = ModelService()
        model = service.get_model(pk)
        if not model:
            return Response({"detail": "Model not found"}, status=404)

        project_dir = "models"
        model_dir = os.path.abspath(os.path.join(project_dir, f"model_{pk}"))
        target_path = os.path.join(model_dir, path)

        if not os.path.abspath(target_path).startswith(os.path.abspath(model_dir)):
            return Response({"detail": "Access denied"}, status=403)
        if not os.path.exists(target_path):
            return Response({"detail": f"File not found: {path}"}, status=404)
        if os.path.isdir(target_path):
            return Response({"detail": "Path is a directory, not a file"}, status=400)

        return FileResponse(target_path, content_type="application/octet-stream", filename=os.path.basename(target_path))