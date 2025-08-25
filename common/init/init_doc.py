# coding=utf-8
import hashlib

from django.urls import re_path, path, URLPattern
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from common.auth import AnonymousAuthentication
from smartdoc.const import CONFIG
from django.utils.translation import gettext_lazy as _


def init_app_doc(application_urlpatterns):
    schema_view = get_schema_view(
        openapi.Info(
            title="测试 API",
            default_version='v1',
            description=_('Intelligent customer service platform,标题配置：common\init\init_doc.py,主路由：smartdoc/urls.py,smartdoc\settings\base.py'),
        ),
        public=True,
        permission_classes=[permissions.AllowAny],
        authentication_classes=[AnonymousAuthentication],
        # url="http://127.0.0.1:8000"  # 添加这个参数来解决 HTTPS/HTTP 混合内容问题
    )
    application_urlpatterns += [
        re_path(r'^doc(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0),
                name='schema-json'),  # 导出
        path('doc/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
        path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    ]


def init_chat_doc(application_urlpatterns, patterns):
    chat_schema_view = get_schema_view(
        openapi.Info(
            title="测试 API：D:\终点\model-develop-standard_dev\common\init\init_doc.py",
            default_version='/chat',
            description=_('Intelligent customer service platform，标题配置：common\init\init_doc.py,主路由：smartdoc/urls.py'),
        ),
        public=True,
        permission_classes=[permissions.AllowAny],
        authentication_classes=[AnonymousAuthentication],
        # url="http://127.0.0.1:8000",  # 添加这个参数来解决 HTTPS/HTTP 混合内容问题
        patterns=[
            URLPattern(pattern='api/' + str(url.pattern), callback=url.callback, default_args=url.default_args,
                       name=url.name)
            for url in patterns if
            url.name is not None and ['application/message', 'application/open',
                                      'application/profile'].__contains__(
                url.name)]
    )

    application_urlpatterns += [
        path('doc/chat/', chat_schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
        path('redoc/chat/', chat_schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    ]


def encrypt(text):
    md5 = hashlib.md5()
    md5.update(text.encode())
    result = md5.hexdigest()
    return result


def get_call(application_urlpatterns, patterns, params, func):
    def run():
        if params['valid']():
            func(*params['get_params'](application_urlpatterns, patterns))

    return run


init_list = [(init_app_doc, {'valid': lambda: CONFIG.get('DOC_PASSWORD') is not None and encrypt(
    CONFIG.get('DOC_PASSWORD')) == 'a10dba64d95b5e6bf23b2b7679040f31',
                             'get_call': get_call,
                             'get_params': lambda application_urlpatterns, patterns: (application_urlpatterns,)}),
            #  (init_chat_doc, {'valid': lambda: CONFIG.get('DOC_PASSWORD') is not None and encrypt(
            #      CONFIG.get('DOC_PASSWORD')) == 'a10dba64d95b5e6bf23b2b7679040f31' or True, 'get_call': get_call,
            #                   'get_params': lambda application_urlpatterns, patterns: (
            #                       application_urlpatterns, patterns)})
                                  ]


def init_doc(application_urlpatterns, patterns):
    for init, params in init_list:
        if params['valid']():
            get_call(application_urlpatterns, patterns, params, init)()
