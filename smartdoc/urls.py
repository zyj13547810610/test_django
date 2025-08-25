"""
URL configuration for apps project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import: from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import: from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function_lib: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os

from django.http import HttpResponse
from django.urls import path, re_path, include
from django.views import static
from rest_framework import status

from common.cache_data.static_resource_cache import get_index_html
from common.constants.cache_code_constants import CacheCodeConstants
from common.init.init_doc import init_doc
from common.response.result import Result
from common.util.cache_util import get_cache
from smartdoc import settings
from smartdoc.conf import PROJECT_DIR
from django.shortcuts import redirect
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from django.contrib import admin


urlpatterns = [
    # path("api/", include("users.urls")),
    path('admin/', admin.site.urls),

    # path("api/", include("setting.urls")),  # 添加setting应用的URL
    # path("api/polls/", include("backend_re.polls.urls")),  # 添加polls应用的URL

    path("api/dashboard/", include("backend_re.dashboard.urls")),  # dashboard
    path("api/settings/", include("backend_re.settings.urls")),  # settings
    path("api/detections/", include("backend_re.detection.urls")),  # detection
    path("api/annotation/", include("backend_re.annotation.urls")),  # annotation
    path("api/models_train/", include("backend_re.modles_train.urls")),  # 添加trains应用的URL
    path("api/cv_operation/", include("backend_re.cv_operation.urls")),  # cv_operation
    path("api/cameras/", include("backend_re.cameras.urls")),  # cameras
]

# urlpatterns = [
#     path("", lambda request: redirect("schema-swagger-ui", permanent=False)),  # 首页跳转到 Swagger UI
#     path("api/", include("users.urls")),
#     path("doc/", SpectacularSwaggerView.as_view(url_name="schema"), name="schema-swagger-ui"),
#     path("redoc/", SpectacularRedocView.as_view(url_name="schema"), name="schema-redoc"),
#     path("doc<format>/", SpectacularAPIView.as_view(), name="schema-json"),
# ]


def pro():
    # 暴露静态主要是swagger资源
    urlpatterns.append(
        re_path(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'),
    )
    # 暴露ui静态资源
    urlpatterns.append(
        re_path(r'^ui/(?P<path>.*)$', static.serve, {'document_root': os.path.join(settings.STATIC_ROOT, "ui")},
                name='ui'),
    )


if not settings.DEBUG:
    pro()


def page_not_found(request, exception):
    """
    页面不存在处理
    """
    if request.path.startswith("/api/"):
        return Result(response_status=status.HTTP_404_NOT_FOUND, code=404, message="找不到接口")
    index_path = os.path.join(PROJECT_DIR, 'apps', "static", 'ui', 'index.html')
    if not os.path.exists(index_path):
        return HttpResponse("页面不存在", status=404)
    content = get_index_html(index_path)
    if request.path.startswith('/ui/chat/'):
        return HttpResponse(content, status=200)
    return HttpResponse(content, status=200, headers={'X-Frame-Options': 'DENY'})


handler404 = page_not_found
# # init_doc(urlpatterns, application_urlpatterns)
# init_doc(application_urlpatterns=urlpatterns,patterns=None)

# 初始化API文档 - 在urlpatterns定义完成后调用
init_doc(urlpatterns, None)
