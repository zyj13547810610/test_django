import os

from django.urls import path

from . import views

app_name = "team"
urlpatterns = [
    path('team/member', views.TeamMember.as_view(), name="team"),
    path('team/member/_batch', views.TeamMember.Batch.as_view()),
    path('team/member/<str:member_id>', views.TeamMember.Operate.as_view(), name='member'),
    path('email_setting', views.SystemSetting.Email.as_view(), name='email_setting'),
    path('valid/<str:valid_type>/<int:valid_count>', views.Valid.as_view())

]

