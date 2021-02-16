#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:16:46 2020

@author: victor
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math

####Parte 1
iris=load_iris()
features=iris.data.T
#Separo en grupos para las etiquetas de la leyenda
groups=[]
for i in range (0,50):
    groups.append("Setosa")
for i in range (0,50):
    groups.append("Versicolour")
for i in range (0,50):
    groups.append("Virginica")


colors={"Setosa":'red', "Versicolour":'green', "Virginica":'blue'}
i=0
#Represento
plt.subplot()
for g in groups:
    plt.scatter(features[2][i], features[3][i], alpha=1.0, s=20, c=colors[g], label=g if i==0 or i==50 or i==100 else "")
    i=i+1
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

plt.legend(loc=2,)
plt.show()

###Parte 2
#Leo la base de datos y tomo las dos primeras columnas
iris_X, iris_y = datasets.load_iris(return_X_y=True)
iris_X=np.delete(iris_X, 0,axis=1)
iris_X=np.delete(iris_X, 0,axis=1)
#Separo para tomar 80/20 de cada flor
#y al juntarlo sea proporcionado
setosa=iris_X[0:50]
versicolour=iris_X[50:100]
virginica=iris_X[100:150]
ysetosa=iris_y[0:50]
yversicolour=iris_y[50:100]
yvirginica=iris_y[100:150]

#Separo con la funcion train_test_split
s_train, s_test, sy_train, sy_test=train_test_split(setosa, ysetosa, test_size=0.2)
ve_train, ve_test, vey_train, vey_test=train_test_split(versicolour, yversicolour,test_size=0.2)
vi_train, vi_test, viy_train, viy_test=train_test_split(virginica, yvirginica, test_size=0.2)

#Finalmente conseguimos los datos de train y test:
X_train=np.concatenate((s_train, ve_train, vi_train), axis=0)
X_test=np.concatenate((s_test, ve_test, vi_test), axis=0)
y_train=np.concatenate((sy_train, vey_train, viy_train), axis=0)
y_test=np.concatenate((sy_test, vey_test, viy_test), axis=0)

#####Parte 3
#Creo las imagenes del vector de los valores equiespaciados
ej3=np.linspace(0.0, 2*math.pi, num=100)
seno=np.sin(ej3)
coseno=np.cos(ej3)
suma=coseno+seno

#Represento cada función en su color
plt.subplot()
plt.plot(ej3, seno, color='black', linestyle='--')
plt.plot(ej3, coseno, color='blue', linestyle='--')
plt.plot(ej3, suma, color='red', linestyle='--')

