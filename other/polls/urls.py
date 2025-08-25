from django.urls import path
from .views.addition import addition_view
from .views.subtraction import subtraction_view
from .views.multiplication import multiplication_view
from .views.division import division_view

app_name = "polls"
urlpatterns = [
    path("add/", addition_view, name="add"),
    path("subtract/", subtraction_view, name="subtract"),
    path("multiply/", multiplication_view, name="multiply"),
    path("divide/", division_view, name="divide"),
] 