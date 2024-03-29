from django.conf.urls import url, include
from rest_framework import routers
from vapi import views
from vapi import urls 

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    url(r'^api/', include(router.urls)),
    url(r'^', include(urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]