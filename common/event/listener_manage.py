# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： listener_manage.py
    @date：2023/10/20 14:01
    @desc:
"""
import logging
import os
import threading
import datetime
import traceback
from typing import List

import django.db.models
from django.db.models import QuerySet
from django.db.models.functions import Substr, Reverse

from common.db.search import native_search, get_dynamics_model, native_update
from common.util.file_util import get_file_content
from common.util.lock import try_lock, un_lock
from common.util.page_utils import page_desc
from smartdoc.conf import PROJECT_DIR
from django.utils.translation import gettext_lazy as _

max_kb_error = logging.getLogger(__file__)
max_kb = logging.getLogger(__file__)
lock = threading.Lock()


class SyncWebDatasetArgs:
    def __init__(self, lock_key: str, url: str, selector: str, handler):
        self.lock_key = lock_key
        self.url = url
        self.selector = selector
        self.handler = handler


class SyncWebDocumentArgs:
    def __init__(self, source_url_list: List[str], selector: str, handler):
        self.source_url_list = source_url_list
        self.selector = selector
        self.handler = handler










class ListenerManagement:
    pass
