from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('plot-graph/', views.plot_graph, name='plot_graph'),
    path('fraud-detect/', views.kafka_producer, name= 'fraud_detect'),
    path('live-view/', views.live_view, name= 'live_view'),
]
