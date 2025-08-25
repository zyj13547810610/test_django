from django.urls import path
from .views.views_detection import *
from .views import addition
app_name = "detection"
urlpatterns = [
    path("api/detections", DetectionListCreateView.as_view()),
    path("api/detections/<int:pk>", DetectionDetailView.as_view()),
    path("api/detections/clear", DetectionClearView.as_view()),
    # path("add/", addition.addition_view, name="add"),
] 