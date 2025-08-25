from django.urls import path , include
from rest_framework.routers import DefaultRouter
from .views.views_cameras import *
from .views import addition

router = DefaultRouter()
router.register(r'', CameraViewSet, basename='cameras')


app_name = "cameras"
urlpatterns = [
    # path("add/", addition.addition_view, name="add"),
    path('', include(router.urls)),  # 自动生成路由
] 