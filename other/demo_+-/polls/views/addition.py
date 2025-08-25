from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from rest_framework import status

@extend_schema(
    parameters=[
        OpenApiParameter("a", OpenApiTypes.FLOAT, OpenApiParameter.QUERY, description="第一个加数"),
        OpenApiParameter("b", OpenApiTypes.FLOAT, OpenApiParameter.QUERY, description="第二个加数"),
    ],
    responses={
        200: {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "result": {"type": "number"},
            },
        }
    },
)
@api_view(["GET"])
def addition_view(request):
    try:
        a = float(request.query_params.get("a", 0))
        b = float(request.query_params.get("b", 0))
    except (TypeError, ValueError):
        return Response({"error": "参数必须是数字"}, status=status.HTTP_400_BAD_REQUEST)
    return Response({"a": a, "b": b, "result": a + b})
