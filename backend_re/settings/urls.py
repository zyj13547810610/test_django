
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views.views_settings import SettingsViewSet  # 注意这里要导入类
router = DefaultRouter()
# router.register(r'settings', SettingsViewSet, basename='settings')
router.register(r'', SettingsViewSet, basename='settings')

app_name = "settings"
urlpatterns = [
# path('api/', include(router.urls)),  # 自动生成路由
path('', include(router.urls)),  # 自动生成路由
] 