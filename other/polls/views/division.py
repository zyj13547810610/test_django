
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework import status
from django.utils.translation import gettext_lazy as _

@swagger_auto_schema(
    method='get',
    operation_summary=_("除法运算"),
    operation_id=_("除法运算"),
    manual_parameters=[
        openapi.Parameter('a', openapi.IN_QUERY, description=_("被除数"), type=openapi.TYPE_NUMBER, required=True),
        openapi.Parameter('b', openapi.IN_QUERY, description=_("除数"), type=openapi.TYPE_NUMBER, required=True),
    ],
    responses={
        200: openapi.Response(
            description=_("计算成功"),
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'a': openapi.Schema(type=openapi.TYPE_NUMBER, description=_("被除数")),
                    'b': openapi.Schema(type=openapi.TYPE_NUMBER, description=_("除数")),
                    'result': openapi.Schema(type=openapi.TYPE_NUMBER, description=_("计算结果")),
                }
            )
        ),
        400: openapi.Response(
            description=_("参数错误"),
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description=_("错误信息")),
                }
            )
        )
    },
    tags=[_("数学运算")]
)
@api_view(["GET"])
def division_view(request):
    """除法运算API"""
    try:
        a = float(request.query_params.get("a", 0))
        b = float(request.query_params.get("b", 1))
    except (TypeError, ValueError):
        return Response({"error": _("参数必须是数字")}, status=status.HTTP_400_BAD_REQUEST)
    
    if b == 0:
        return Response({"error": _("除数不能为0")}, status=status.HTTP_400_BAD_REQUEST)
    
    result = a / b
    return Response({
        "a": a, 
        "b": b, 
        "result": result,
        "operation": "division"
    }) 