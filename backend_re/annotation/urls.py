from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views.views_annotation import AnnotationViewSet  # 注意这里要导入类

from .views import addition
router = DefaultRouter()
router.register(r'', AnnotationViewSet, basename='annotation')
app_name = "annotation"
urlpatterns = [
    # path('api/', include(router.urls)),  # 自动生成路由
    path('', include(router.urls)),  # 自动生成路由
    path("add/", addition.addition_view, name="add"),
] 