from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('homepage/', views.home, name='homedata'),

    path('manage_data/', views.managedata, name='mndata'),
    path('manage_genus/', views.managegenus, name='genusdata'),
    path('manage_species/', views.managespeci, name='specidata'),
    path('manage_info/', views.manageinfo, name='infodata'),

    path('genus_add/', views.addgenus, name='addgenu'),
    path('genus_delete/<genu_id>/', views.genusdelete, name='genusde'),
    path('genus_update/<int:gn_id>/', views.genusupdate, name='updatege'),
    path('genus_search/', views.genussearch, name='genusearch'),

    path('species_add/', views.addspecies, name='addspec'),
    path('species_delete/<spec_id>/', views.deletespecies, name='speciesdelete'),
    path('species_update/<int:spec_id>/', views.updatespecies, name='speciesupdate'),
    path('species_search/', views.searchspecies, name='speciessearch'),

    path('classify_page/', views.predictplant, name='classify'),

    path('login_form/', views.adminlogin, name='formlogin'),
]
