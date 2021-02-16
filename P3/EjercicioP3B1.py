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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron, Lasso, LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest
# Fijamos la semilla
np.random.seed(1)

def readData(name):
	# Leemos los ficheros	
    data = np.loadtxt(name, delimiter=",")
    return data[:,:-1], data[:,-1]

# 70% training 30% test
# Lectura de los datos de entrenamiento
x_train, y_train = readData('./datos/optdigits.tra')
# Lectura de los datos para el test
x_test, y_test = readData("./datos/optdigits.tes")

#Preprocesado de los datos
preproc=[("var", VarianceThreshold(0.01)),
            ("poly", PolynomialFeatures(2)),
            ("standardize", StandardScaler())]


pipe=Pipeline(preproc + [('lr', LogisticRegression())])
#CV con varios modelos
params_grid=[
        {"lr":[LogisticRegression(penalty='l2',max_iter=500)],
                "lr__C":np.logspace(-4,4,5),
                "lr__solver":['lbfgs','newton-cg'],
                "poly__degree":[1,2]},
        {"lr":[Perceptron(random_state=1)],
               "lr__penalty":['l1','l2'],
               "lr__alpha":np.logspace(-6,1,5),
               "poly__degree":[1,2]},
        {"lr":[SGDClassifier(max_iter=500, tol=1e-3)],
               "lr__loss":['log','hinge'],
               "lr__penalty":['l1','l2']}]

#Mostramos el mejor usando de scoring (métrica) accuracy
best_lr=GridSearchCV(pipe,params_grid, scoring='accuracy', cv=5, n_jobs=-1)
best_lr.fit(x_train,y_train)
print("\n ----- Sin Lasso y con preprocesado ------\n")
print("Parámetros del mejor clasificador:\n{}".format(best_lr.best_params_))
print("Accuracy en CV: {:0.3f}%".format(100.0 * best_lr.best_score_))
print("Accuracy en training: {:0.3f}%".format(
        100.0 * best_lr.score(x_train, y_train)))
print("Accuracy en test: {:0.3f}%".format(
        100.0 * best_lr.score(x_test, y_test)))


#Con regularización
#Preprocesado con lasso
preprocReg=[("var", VarianceThreshold(0.01)),
            ("regLasso",SelectFromModel(Lasso())),
            ("poly", PolynomialFeatures(2)),
            ("standardize", StandardScaler())]

pipe=Pipeline(preprocReg + [('lr', LogisticRegression())])
#Hacemos CV con varios modelos
params_grid=[
        {"lr":[LogisticRegression(penalty='l2',max_iter=500)],
                "lr__C":np.logspace(-4,4,5),
                "lr__solver":['lbfgs','newton-cg'],
                "poly__degree":[1,2]},
        {"lr":[Perceptron(random_state=1)],
               "lr__penalty":['l1','l2'],
               "lr__alpha":np.logspace(-6,1,5),
               "poly__degree":[1,2]},
        {"lr":[SGDClassifier(max_iter=500, tol=1e-3)],
               "lr__loss":['log','hinge'],
               "lr__penalty":['l1','l2']}]

#Mostramos el mejor usando de scoring (métrica) accuracy
best_lr=GridSearchCV(pipe,params_grid, scoring='accuracy', cv=5, n_jobs=-1)
best_lr.fit(x_train,y_train)
print("\n ----- Con Lasso y con preprocesado ------\n")
print("Parámetros del mejor clasificador:\n{}".format(best_lr.best_params_))
print("Accuracy en CV: {:0.3f}%".format(100.0 * best_lr.best_score_))
print("Accuracy en training: {:0.3f}%".format(
        100.0 * best_lr.score(x_train, y_train)))
print("Accuracy en test: {:0.3f}%".format(
        100.0 * best_lr.score(x_test, y_test)))


