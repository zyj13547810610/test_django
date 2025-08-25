from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError
from django.shortcuts import get_object_or_404
from datetime import datetime
from .base import (
    DetectionCreateSerializer,
    DetectionResponseSerializer,
    PaginatedDetectionResponseSerializer
)
from backend_re.backend.models import DetectionStatus
from ..services import DetectionService
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


class DetectionListCreateView(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter("page", openapi.IN_QUERY, description="页码", type=openapi.TYPE_INTEGER),
            openapi.Parameter("per_page", openapi.IN_QUERY, description="每页数量", type=openapi.TYPE_INTEGER),
            openapi.Parameter("search_text", openapi.IN_QUERY, description="搜索文本", type=openapi.TYPE_STRING),
            openapi.Parameter("status", openapi.IN_QUERY, description="状态", type=openapi.TYPE_STRING),
            openapi.Parameter("date_from", openapi.IN_QUERY, description="起始日期 YYYY-MM-DD", type=openapi.TYPE_STRING),
            openapi.Parameter("date_to", openapi.IN_QUERY, description="结束日期 YYYY-MM-DD", type=openapi.TYPE_STRING),
            openapi.Parameter("device_id", openapi.IN_QUERY, description="设备ID", type=openapi.TYPE_INTEGER),
            openapi.Parameter("confidence_min", openapi.IN_QUERY, description="最小置信度", type=openapi.TYPE_NUMBER),
            openapi.Parameter("confidence_max", openapi.IN_QUERY, description="最大置信度", type=openapi.TYPE_NUMBER),
        ]
    )
        
    def get(self, request):
        """获取检测记录列表，带分页和过滤"""
        page = int(request.query_params.get("page", 1))
        per_page = int(request.query_params.get("per_page", 10))
        search_text = request.query_params.get("search_text")
        status_param = request.query_params.get("status")
        date_from = request.query_params.get("date_from")
        date_to = request.query_params.get("date_to")
        device_id = request.query_params.get("device_id")
        confidence_min = request.query_params.get("confidence_min")
        confidence_max = request.query_params.get("confidence_max")

        # 日期解析
        def parse_date(d):
            if d:
                try:
                    return datetime.strptime(d, "%Y-%m-%d")
                except ValueError:
                    raise ValidationError(f"无效的日期格式: {d}，应为YYYY-MM-DD")
            return None

        date_from_dt = parse_date(date_from)
        date_to_dt = parse_date(date_to)

        detection_service = DetectionService()
        result = detection_service.list_detections(
            page=page,
            per_page=per_page,
            search_text=search_text,
            status=status_param,
            date_from=date_from_dt,
            date_to=date_to_dt,
            device_id=device_id,
            confidence_min=confidence_min,
            confidence_max=confidence_max
        )

        serializer = PaginatedDetectionResponseSerializer({
            "items": result["items"],
            "total": result["total"],
            "page": result["page"],
            "per_page": result["per_page"],
            "pages": result["pages"]
        })
        return Response(serializer.data)
    
    @swagger_auto_schema(request_body=DetectionCreateSerializer)
    def post(self, request):
        """创建新的检测记录"""
        serializer = DetectionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # 拷贝 validated_data，避免重复传 status
        data = serializer.validated_data.copy()

        # 处理 status 字段
        status_map = {
            "pass": DetectionStatus.PASS,
            "合格": DetectionStatus.PASS,
            "fail": DetectionStatus.FAIL,
            "不合格": DetectionStatus.FAIL
        }
        raw_status = data.get("status")
        if raw_status:
            data["status"] = status_map.get(raw_status.lower(), DetectionStatus.UNKNOWN)
        else:
            data["status"] = DetectionStatus.UNKNOWN

        # 创建检测记录
        detection_service = DetectionService()
        detection = detection_service.create_detection(**data)

        return Response(DetectionResponseSerializer(detection).data, status=status.HTTP_201_CREATED)


class DetectionDetailView(APIView):

    def get(self, request, pk):
        """获取单个检测记录详情"""
        detection_service = DetectionService()
        detection = detection_service.get_detection(pk)
        if not detection:
            return Response({"detail": "检测记录不存在"}, status=status.HTTP_404_NOT_FOUND)
        return Response(DetectionResponseSerializer(detection).data)

    def delete(self, request, pk):
        """删除检测记录"""
        detection_service = DetectionService()
        success = detection_service.delete_detection(pk)
        if not success:
            return Response({"detail": "检测记录不存在"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"message": "检测记录已删除"})


class DetectionClearView(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter("before_date", openapi.IN_QUERY, description="清除此日期之前的数据 YYYY-MM-DD", type=openapi.TYPE_STRING),
            openapi.Parameter("status", openapi.IN_QUERY, description="状态过滤", type=openapi.TYPE_STRING),
            openapi.Parameter("device_id", openapi.IN_QUERY, description="设备ID过滤", type=openapi.TYPE_INTEGER),
        ]
    )
    def delete(self, request):
        """清除满足条件的检测记录"""
        before_date = request.query_params.get("before_date")
        status_param = request.query_params.get("status")
        device_id = request.query_params.get("device_id")

        before_date_dt = None
        if before_date:
            try:
                before_date_dt = datetime.strptime(before_date, "%Y-%m-%d")
            except ValueError:
                raise ValidationError(f"无效的日期格式: {before_date}，应为YYYY-MM-DD")

        status_enum = None
        if status_param:
            status_map = {
                "pass": DetectionStatus.PASS,
                "合格": DetectionStatus.PASS,
                "fail": DetectionStatus.FAIL,
                "不合格": DetectionStatus.FAIL,
                "unknown": DetectionStatus.UNKNOWN,
                "未知": DetectionStatus.UNKNOWN
            }
            status_enum = status_map.get(status_param.lower())

        detection_service = DetectionService()
        count = detection_service.clear_detections(
            before_date=before_date_dt,
            status=status_enum,
            device_id=device_id
        )

        return Response({"message": f"已清除 {count} 条检测记录"})
