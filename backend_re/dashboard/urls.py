from django.urls import path
# from .views.dashboard import *
from .views import views_dashboard
from .views import addition



app_name = "dashboard"
urlpatterns = [
    # path("add/", addition.addition_view, name="add"),
    # path("subtract/", subtraction_view, name="subtract"),
    # path("multiply/", multiplication_view, name="multiply"),
    # path("divide/", division_view, name="divide"),
    path("system-status/", views_dashboard.get_system_status, name="system_status"),
    path("detection-stats/", views_dashboard.get_detection_stats, name="detection_stats"),
    path("defect-stats/", views_dashboard.get_defect_stats, name="defect_stats"),
    path("accuracy/", views_dashboard.get_accuracy, name="accuracy"),
    path("alerts/", views_dashboard.get_recent_alerts, name="recent_alerts"),
    path("alerts/create/", views_dashboard.create_alert, name="create_alert"),
    path("alerts/<int:alert_id>/read/", views_dashboard.mark_alert_as_read, name="mark_alert_as_read"),
    path("device-status/", views_dashboard.get_device_status_summary, name="device_status_summary"),
    path("detection-trends/", views_dashboard.get_daily_detection_trends, name="daily_detection_trends"),   

] 