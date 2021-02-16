#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:36 2020

@author: victor
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as m

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Perceptron, Lasso, LogisticRegression, LinearRegression, SGDRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest
from sklearn.impute import SimpleImputer
# Fijamos la semilla
np.random.seed(1)

def readData(name):
	# Leemos los ficheros	
    data = np.genfromtxt(name, delimiter = ",", dtype = np.double)
    return data[:, :-1], data[:, -1]


# Lectura de los datos de entrenamiento
x, y = readData('./datos/communities.data')
#Borramos las 5 primeras columnas pues no aportan información
x=np.delete(x, 1, axis=1)
x=np.delete(x, 1, axis=1)
x=np.delete(x, 1, axis=1)
x=np.delete(x, 1, axis=1)
x=np.delete(x, 1, axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, shuffle=True)



#Preprocesado de los datos
preproc=[("missing", SimpleImputer()),
         ("var", VarianceThreshold(0.01)),
            ("poly", PolynomialFeatures(1)),
            ("standardize", StandardScaler())]


pipe=Pipeline(preproc + [('model', SGDRegressor())])

params_grid=[
        {"model":[SGDRegressor(max_iter=500)],
               "model__loss":['huber', 'squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
               "model__penalty":['l1','l2'],
               "model__alpha":np.logspace(-5,5,5),
               "poly__degree":[1,2]},
        {"model":[LinearRegression()],
               "poly__degree":[1,2]},
        {"model":[Ridge()],
               "poly__degree":[1,2],
               "model__alpha":np.logspace(-5,5,5)},
        {"model":[Lasso()],
               "poly__degree":[1,2],
               "model__alpha":np.logspace(-5,5,5)}]

#Mostramos el mejor sin regularizacion
best_model=GridSearchCV(pipe,params_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
best_model.fit(x_train,y_train)
print("Parámetros del mejor clasificador:\n{}".format(best_model.best_params_))
print("Error en CV: {:0.3f}%".format(100.0 * best_model.best_score_))
print("Error en training: {:0.3f}%".format(
        100.0 * best_model.score(x_train, y_train)))
print("Error en test: {:0.3f}%".format(
        100.0 * best_model.score(x_test, y_test)))

