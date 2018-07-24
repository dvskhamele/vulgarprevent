from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^train/$', views.train, name='train'),
    url(r'^predict/$', views.predict, name='predict'),
]

