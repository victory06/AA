#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:37:52 2020

@author: victor
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from mpl_toolkits import mplot3d

#BONUS: Metodo de Newton con los experimentos del Ejercicio 3
#a)
#Definimos la función y sus derivadas para la matriz Hessiana
def fun(w): 
    x=w[0]
    y=w[1]
    return (x-2)**2+2*(y+2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def fx(w):
    x=w[0]
    y=w[1]
    return 2*(x-2)+4*math.pi*math.cos(2*math.pi*x)*math.sin(2*math.pi*y)

def fy(w):
    x=w[0]
    y=w[1]
    return 4*(y+2)+4*math.pi*math.sin(2*math.pi*x)*math.cos(2*math.pi*y)

def df(w):
    df=[]
    df.append(fx(w))
    df.append(fy(w))
    return df

def dxy(w):
    x=w[0]
    y=w[1]
    return 8*(math.pi**2)*math.cos(2*math.pi*x)*math.cos(2*math.pi*y)

def d2x(w):
    x=w[0]
    y=w[1]
    return 2-8*(math.pi**2)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

def d2y(w):
    x=w[0]
    y=w[1]
    return 4-8*(math.pi**2)*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

def dyx(w):
    x=w[0]
    y=w[1]
    return 8*(math.pi**2)*math.cos(2*math.pi*x)*math.cos(2*math.pi*y)

#Versión diapositivas teoría
def Newton_grafica(x0,y0, fun, df, d2x, d2y, dxy, dyx, lr, max_iters = 50):
    graf=[]
    it=0
    w=[]
    w.append(x0)
    w.append(y0)
    while it<max_iters:
        hess=[]
        f1=[d2x(w),dxy(w)]
        f2=[dyx(w),d2y(w)]
        hess.append(np.array([f1,f2]))
        grad=df(w)
        graf.append(fun(w))
        mat=np.linalg.inv(hess).dot(grad) #devuelve una matriz, pasamos a vector
        mat2=[mat[0][0], mat[0][1]]
        j=[lr*mat2[0],lr*mat2[1]]
        w[0]=w[0]-j[0]
        w[1]=w[1]-j[1]
        it+=1
    
    plt.plot(range(0,max_iters), graf, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()	
    return w

#Versión optimización al mínimo
def Newton_grafica2(x0,y0, fun, df, d2x, d2y, dxy, dyx, lr, max_iters = 50):
    graf=[]
    it=0
    w=[]
    w.append(x0)
    w.append(y0)
    while it<max_iters:
        graf.append(fun(w))
        w[0]=w[0]-fun(w)/(df(w)[0])
        w[1]=w[1]-fun(w)/(df(w)[1])
        it+=1
    
    plt.plot(range(0,max_iters), graf, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()	
    return w


print ('Resultados BONUS a)')


x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(x, y)
Z = fun([X, Y])
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.view_init(0, 35)
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('Representación de f');
ax.contour3D(X, Y, Z, 50, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()

print ('\nGrafica con learning rate igual a 0.01')
w=Newton_grafica(1.0, -1.0, fun, df, d2x, d2y, dxy, dyx, 0.01)
print ('\nGrafica con learning rate igual a 0.1')
w=Newton_grafica(1.0, -1.0, fun, df, d2x, d2y, dxy, dyx, 0.1)
input("\n--- Pulsar tecla para continuar ---\n")

# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

print ('Punto de inicio: (2.1, -2.1)\n')
w=Newton_grafica(2.1, -2.1, fun, df, d2x, d2y, dxy, dyx, 0.1)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor maximo: ',fun(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w=Newton_grafica(3.0, -3.0, fun, df, d2x, d2y, dxy, dyx, 0.1)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor maximo: ',fun(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w=Newton_grafica(1.5, 1.5, fun, df, d2x, d2y, dxy, dyx, 0.01)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor maximo: ',fun(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w=Newton_grafica(1.0, -1.0, fun, df, d2x, d2y, dxy, dyx, 0.1)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor maximo: ',fun(w))

