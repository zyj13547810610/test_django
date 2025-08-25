from django.urls import path , include
from rest_framework.routers import DefaultRouter
from .views.views_cv_operation import *
from .views import addition


router = DefaultRouter()
router.register(r'', CVOperationViewSet, basename='cv_operation')


app_name = "cv_operation"
urlpatterns = [
    # path("add/", addition.addition_view, name="add"),
    path('', include(router.urls)),  # 自动生成路由
] 