"""
URL configuration for mywebsite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path
from pages.views import home,dc,csk,mi,srh,rcb,rr,kkr,gt,lsg,pbks,prediction_view,predict,select_teams,submit_players,batting_selection,predict_target,predict_win,contact_us

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('dc/',dc,name='dc'),
    path('csk/',csk,name='csk'),
    path('gt/',gt,name='gt'),
    path('kkr/',kkr,name='kkr'),
    path('lsg/',lsg,name='lsg'),
    path('mi/',mi,name='mi'),
    path('pbks/',pbks,name='pbks'),
    path('rcb/',rcb,name='rcb'),
    path('rr/',rr,name='rr'),
    path('srh/',srh,name='srh'),
    path('prediction_view/',prediction_view, name='prediction_view'),
    path('predict/',predict, name='predict'),
    path('select_teams/',select_teams, name='select_teams'),
    path('submit_players/',submit_players, name='submit_players'),
    path('batting_selection/',batting_selection, name='batting_selection'),
    path('predict_target/',predict_target, name='predict_target'),
    path('predict_win/',predict_win, name='predict_win'),
    path('contact_us/',contact_us, name='contact_us'),
]
