from ..services import AnnotationService
from backend_re.backend.models import DatasetType,DATASET_TYPE_CHOICES,Annotation
from typing import List, Dict, Any
from datetime import datetime
import os
from pydantic import Field, validator

from rest_framework import serializers
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.parsers import MultiPartParser, FormParser



class AnnotationResponseSerializer(serializers.ModelSerializer):
    """标注数据基础模型"""
    annotations = serializers.SerializerMethodField()

    class Meta:
        model = Annotation
        fields = ['annotations']

    def get_annotations(self, obj):
        # obj.data 里存的是 JSON 字典
        return obj.data.get('annotations', [])
    
    
class DatasetResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField(help_text="数据集ID")
    name = serializers.CharField(min_length=1, max_length=100, help_text="数据集名称")
    type = serializers.ChoiceField(choices=DATASET_TYPE_CHOICES, help_text="数据集类型")
    imageCount = serializers.IntegerField(help_text="图片数量")
    createdAt = serializers.DateTimeField(help_text="创建时间")
    updatedAt = serializers.DateTimeField(help_text="更新时间")
    class Meta:
        swagger_schema_fields = {
            "example": {
                "id": 1,
                "name": "文本区域标注数据集",
                "type": "TEXT_REGION",
                "imageCount": 100,
                "createdAt": "2024-03-22T10:00:00",
                "updatedAt": "2024-03-22T10:00:00"
            }
        }



class ImageResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField(help_text="图片ID")
    filename = serializers.CharField(help_text="文件名")
    url = serializers.CharField(help_text="图片URL")
    datasetId = serializers.IntegerField(help_text="所属数据集ID")
    createdAt = serializers.DateTimeField(help_text="创建时间")
    isAnnotated = serializers.BooleanField(default=False, help_text="是否已标注")

    class Meta:
        swagger_schema_fields = {
            "example": {
                "id": 1,
                "filename": "example.jpg",
                "url": "/uploads/1/example.jpg",
                "datasetId": 1,
                "createdAt": "2024-03-22T10:00:00",
                "isAnnotated": False
            }
        }

# --- 请求模型 ---
class CreateDatasetRequestSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=1, max_length=100, help_text="数据集名称")
    type = serializers.ChoiceField(choices=DATASET_TYPE_CHOICES, help_text="数据集类型")

    def validate_name(self, v):
        value = v.strip()
        if not value:
            raise serializers.ValidationError("数据集名称不能为空")
        return value

class UpdateDatasetRequestSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=1, max_length=100, help_text="数据集名称")

    def validate_name(self, v):
        value = v.strip()
        if not value:
            raise serializers.ValidationError("数据集名称不能为空")
        return value

class AnnotationRequestSerializer(serializers.Serializer):
    annotations = serializers.ListField(child=serializers.DictField(),help_text="标注数据")

    def validate_annotations(self, v):
        if not isinstance(v, list):
            raise serializers.ValidationError("标注数据必须是列表")
        return v

# --- 路由处理函数 ---

class AnnotationViewSet(viewsets.ViewSet):
    # parser_classes = (MultiPartParser, FormParser)
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.service=AnnotationService()
        
        
    @swagger_auto_schema(
        operation_description= "获取所有数据集",
        responses={
                200: DatasetResponseSerializer(many=True),
                400: openapi.Schema(  # 失败时返回的错误格式
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                    }
                )
            }
    )
    @action(detail=False, methods=["get"],url_path='get_datasets')
    def get_datasets(self,request):
        """获取所有数据集"""
        datasets = self.service.get_datasets()
        serializer=DatasetResponseSerializer([
            {            
                "id": dataset.id,
                "name": dataset.name,
                "type": dataset.type,
                # "imageCount": len(dataset.images),
                "imageCount":  dataset.images.count() if dataset.images else 0 ,
                "createdAt": dataset.created_at, #这里 字典的 key 就是 createdAt 和 updatedAt，Serializer 直接拿 dict 的 key，就不会去找模型字段。类比cv_operation部分
                "updatedAt": dataset.updated_at
            } for dataset in datasets
        ],many=True )
        return Response(serializer.data)
        
    

    @swagger_auto_schema(
        operation_description= "创建新数据集",
        request_body=CreateDatasetRequestSerializer,
        responses={
                201: DatasetResponseSerializer,  # 成功时返回的数据结构
                400: openapi.Schema(  # 失败时返回的错误格式
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                    }
                )
            }
    )
    @action(detail=False,methods=['post'],url_path='create_datasets')
    def create_datasets(self,request):
        """创建新数据集"""
        req_serializer = CreateDatasetRequestSerializer(data=request.data)
        req_serializer.is_valid(raise_exception=True)
        try:
            dataset = self.service.create_dataset(req_serializer.validated_data['name'],req_serializer.validated_data['type'])
            image_count = dataset.images.count() if dataset.images else 0
            resp_serializer = DatasetResponseSerializer({
                'id':dataset.id,
                "name": dataset.name,
                "type": dataset.type,
                # "imageCount": len(dataset.images),
                "imageCount": image_count,
                "createdAt": dataset.created_at,
                "updatedAt": dataset.updated_at
            })
            return Response(resp_serializer.data,status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'detail':f"Failed to create dataset: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)
            
    @swagger_auto_schema(
        operation_description= "重命名数据集",
        request_body=UpdateDatasetRequestSerializer,
        responses={
                201: DatasetResponseSerializer,  # 成功时返回的数据结构
                400: openapi.Schema(  # 失败时返回的错误格式
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                    }
                )
            }
    )
    @action(detail=True,methods=['put'],url_path='rename_dataset')      
    def rename_dataset(self,request,pk=None):
        """重命名数据集"""
        req_serializer = UpdateDatasetRequestSerializer(data=request.data)
        req_serializer.is_valid(raise_exception=True)
        try:
            dataset=self.service.rename_dataset(pk,req_serializer.validated_data['name'])
            resp_serializer=DatasetResponseSerializer({
                            'id':dataset.id,
                            "name": dataset.name,
                            "type": dataset.type,
                            # "imageCount": len(dataset.images),
                            "imageCount": dataset.images.count() if dataset.images else 0,
                            "createdAt": dataset.created_at,
                            "updatedAt": dataset.updated_at
            })
            return Response(resp_serializer.data,status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'detail':f"Failed to create dataset: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)
            
            
    @swagger_auto_schema(
        operation_description= "删除数据集",
        responses={
                201: DatasetResponseSerializer,  # 成功时返回的数据结构
                400: openapi.Schema(  # 失败时返回的错误格式
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                    }
                )
            }
    )
    @action(detail=True,methods=['delete'],url_path='delete_dataset')  
    def delete_dataset(self,request,pk=None):
        """删除数据集"""
        success = self.service.delete_dataset(pk)
        if not success:
            return Response({"detail": "dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"message": "dataset deleted successfully"})


    @action(detail=True,methods=['get'],url_path='get_dataset_images')  
    def get_dataset_images(self,request,pk=None):
        """获取数据集下的所有图片"""
        try:
            images=self.service.get_dataset_images(pk)
            serializer=ImageResponseSerializer([
            {            
                "id": image.id,
                "filename": image.filename,
                "url": image.url,
                "createdAt": image.created_at,
                "isAnnotated": image.is_annotated,
                "datasetId": image.dataset_id
            } for image in images
        ],many=True )
            return Response(serializer.data)
        except Exception as e:
            return Response({'detail':f"Failed to create dataset: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)
            
            
    @swagger_auto_schema(
    operation_description="上传图片到数据集",
    manual_parameters=[
        openapi.Parameter(
            name='files',
            in_=openapi.IN_FORM,
            type=openapi.TYPE_FILE,
            description='上传的图片文件',
            required=True
        )
    ],
    responses={
        200: ImageResponseSerializer(many=True),
        400: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息，例如：文件保存失败或数据集不存在")
                }   
            )
        }
    )
    @action(detail=True,methods=['post'],url_path='upload_images', parser_classes=[MultiPartParser, FormParser])  
    def upload_images(self,request,pk=None):
        """上传图片到数据集"""
        uploaded_images = []
        try:
            upload_dir = os.path.join("uploads", str(pk))
            os.makedirs(upload_dir, exist_ok=True)
                   
            for file in request.FILES.getlist('files'):
                # 生成安全的文件名
                filename = file.name
                file_path = os.path.join(upload_dir, filename)
                
                # 保存文件
                try:
                    with open(file_path, "wb") as f:
                        for chunk in file.chunks():
                            f.write(chunk)
                except Exception as e:
                        return Response({'detail': f"保存文件 {filename} 失败: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
                                
                # 保存到数据库
                try:
                    url = f"/uploads/{pk}/{filename}"
                    image = self.service.create_image(
                        dataset_id=pk,
                        filename=filename,
                        url=url
                    )
                    uploaded_images.append({
                            'id': image.id,
                            'filename': image.filename,
                            'url': url,
                            'datasetId': image.dataset_id,
                            'createdAt': image.created_at,
                            'isAnnotated': image.is_annotated
                            })
                except Exception as e:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                return Response({'detail': f"保存图片 {filename} 到数据库失败: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

                serializer = ImageResponseSerializer(uploaded_images, many=True)
                return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'detail': f"上传图片失败: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True,methods=['delete'],url_path='delete_image')  
    def delete_image(self,request,pk=None):
        """删除图片"""
        self.service.delete_image(pk)
        return Response({"status": "success"})


    @swagger_auto_schema(
        operation_description="获取图片的标注数据",
        responses={
            200: AnnotationResponseSerializer,
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                }
            )
        }
    )
    @action(detail=True,methods=['get'],url_path='get_image_annotation')  
    def get_image_annotation(self,request,pk=None):
        """获取图片的标注数据"""
        try:
            annotation=self.service.get_image_annotation(pk)
            if annotation:
                return Response(AnnotationRequestSerializer(annotation).data)
            return Response(AnnotationRequestSerializer())
        except Exception as e:
                return Response({'detail':f"Failed to get annotation: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST)


    @swagger_auto_schema(
        operation_description="创建标注数据",
        request_body=AnnotationRequestSerializer, # 这里会生成 JSON 输入框
        responses={
            200: AnnotationResponseSerializer,
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "detail": openapi.Schema(type=openapi.TYPE_STRING, description="错误信息")
                }
            )
        }
    )
    @action(detail=True,methods=['post'],url_path='save_annotation')  
    def save_annotation(self,request,pk=None):
        """创建标注数据"""
        req_serializer = AnnotationRequestSerializer(data=request.data)
        req_serializer.is_valid(raise_exception=True)
        try:
            annotation = self.service.save_annotation(pk, req_serializer.validated_data)
            resp_serializer = AnnotationResponseSerializer(annotation)
            return Response(resp_serializer.data)
        except Exception as e:
                return Response({'detail':f"Failed to create annotation: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST)
