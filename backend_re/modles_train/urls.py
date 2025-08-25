from django.urls import path , include
from rest_framework.routers import DefaultRouter
from .views.views_models import ModelViewSet
from .views import addition

router = DefaultRouter()
router.register(r'', ModelViewSet, basename='models')

app_name = "models"
urlpatterns = [
    # path("add/", addition.addition_view, name="add"),
    path('', include(router.urls)),  # 自动生成路由
] 