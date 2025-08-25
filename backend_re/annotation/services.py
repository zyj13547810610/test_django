from ..backend.models import Dataset, Image, Annotation, DatasetType
# from datetime import datetime
from django.utils import timezone
from rest_framework.exceptions import APIException
from typing import List, Optional
import os

class AnnotationService:
    def __init__(self):
        pass
    def get_datasets(self) -> List[Dataset]:
        return Dataset.objects.all()

    def create_dataset(self, name: str, type: DatasetType = DatasetType.TEXT_REGION) -> Dataset:
        try:
            dataset = Dataset(
                name=name,
                type=type
            )
            dataset.save()
            return dataset
        except Exception as e:
            raise e

    def rename_dataset(self, dataset_id: int, new_name: str) -> Dataset:
        dataset = Dataset.objects.filter(id=dataset_id).first()
        if not dataset:
            raise APIException(detail="Dataset not found", code=404)
        dataset.name = new_name
        dataset.updated_at = timezone.now() 
        dataset.save()
        return dataset

    def delete_dataset(self, dataset_id: int):
        dataset = Dataset.objects.filter(id=dataset_id).first()
        if not dataset:
            raise APIException( detail="Dataset not found",code=404)
        dataset.delete()
        return True

    def get_dataset_images(self, dataset_id: int) -> List[Image]:
        dataset = Dataset.objects.filter(id=dataset_id).first()
        if not dataset:
            raise APIException(detail="Dataset not found", code=404)
        return dataset.images.all()

    def create_image(self, dataset_id: int, filename: str, url: str) -> Image:
        """创建新图片"""
        try:
            # 检查数据集是否存在
            dataset = Dataset.objects.filter(id=dataset_id).first()
            if not dataset:
                raise APIException(detail="Dataset not found", code=404)

            image = Image(
                filename=filename,
                url=url,
                dataset_id=dataset_id,
                created_at=timezone.now(),
                is_annotated=False
            )
            image.save()
            return image
        except Exception as e:
            raise e

    def delete_image(self, image_id: int):
        """删除图片"""
        image = Image.objects.filter(id=image_id).first()
        if not image:
            raise APIException(detail="Image not found", code=404)
        
        try:
            # 从 url 中获取文件路径
            # url 格式: /uploads/{dataset_id}/{filename}
            file_path = os.path.join(".", image.url.lstrip('/'))
            
            # 删除物理文件
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 删除数据库记录
            image.delete()
            
        except Exception as e:
            # raise e
            raise APIException(detail=f"Failed to delete image: {str(e)}",code=500,
            )

    def get_image_annotation(self, image_id: int) -> Optional[dict]:
        try:
            # Changed to get the single annotation for the image
            ann = Annotation.objects.filter(image_id=image_id).first()
            if not ann:
                return None
            
            return ann.data
        except Exception as e:
            raise e

    def save_annotation(self, image_id: int, annotation_data: dict) -> Annotation:
        try:
            # Check if image exists
            image = Image.objects.filter(id=image_id).first()
            if not image:
                raise APIException(detail="Image not found", code=404)
            
            # Check if annotation already exists for this image
            existing_annotation = Annotation.objects.filter(image_id = image_id).first()
            
            if existing_annotation:
                # Update existing annotation instead of creating a new one
                existing_annotation.data = annotation_data
                existing_annotation.updated_at = timezone.now()
                existing_annotation.save()
                
                # Update image annotation status
                if len(annotation_data.get('annotations', [])) == 0:
                    image.is_annotated = False
                else:
                    image.is_annotated = True
                image.save()
                
                return existing_annotation
            
            # Create new annotation if none exists
            annotation = Annotation(
                image_id=image_id,
                data=annotation_data
            )
            annotation.save()

            # Update image annotation status
            image.is_annotated = True
            image.save()

            return annotation
        except Exception as e:
            raise e
