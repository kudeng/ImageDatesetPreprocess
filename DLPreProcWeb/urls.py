"""DLPreProcWeb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.conf.urls import *
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.urls import path
from django.http import HttpResponse, HttpResponseRedirect
from process import views


def front(request):
    return HttpResponseRedirect("index.html")


urlpatterns = [
    # url(r'^admin /', include(admin.site.urls)),
    # url(r'^show_pages /', include('process.urls')),
    # url(r'^static/(?P.*)$', 'django.views.static.server', {'document_root': settings.STATIC_ROOT}, name='static'),
    # path('', login_required(views.index)),
    path('', front),
    path('sign-up.html', views.djregist, name="signup"),
    path('forgot.html', views.forgot, name="forget"),
    path('index.html', views.djlogin, name="login"),
    path('profile.html', login_required(views.profile), name="profile"),
    path('table.html', login_required(views.table), name="table"),
    path('task.html', login_required(views.task), name="task"),
    path('help.html', login_required(views.helpme), name="help"),
    path("logout.html", login_required(views.djlogout), name="logout"),
    path("download/<file_name>", login_required(views.download), name="download"),
    path("preview.html", login_required(views.preview), name="preview"),
]
