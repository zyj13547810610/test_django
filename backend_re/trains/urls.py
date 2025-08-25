from django.urls import path
from .views.views_trains import *
from .views import addition
app_name = "trains"
urlpatterns = [
    path("add/", addition.addition_view, name="add"),
] 