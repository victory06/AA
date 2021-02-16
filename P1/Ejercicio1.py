#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:57:04 2020

@author: victor
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################



#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla

def E(w): 
    u=w[0]
    v=w[1]
    return (u*np.exp(v)-2*v*np.exp(-u))**2
			 
# Derivada parcial de E respecto de u
def Eu(w):
    u=w[0]
    v=w[1]
    return 2*(u*math.exp(v)-2*v*math.exp(-u))*(math.exp(v)+2*v*math.exp(-u))

# Derivada parcial de E respecto de v
def Ev(w):
    u=w[0]
    v=w[1]
    return 2*(u*math.exp(v)-2*math.exp(-u))*(u*math.exp(v)-2*v*math.exp(-u))
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

#Algoritmo del gradiente descendiente

##Apartado 1.
def gd(w, lr, grad_fun, fun, epsilon, max_iters = 400 ):	
    it=0
    while fun(w)>=epsilon and it<max_iters:
        w=w-lr*grad_fun(w)
        it+=1
    return w, it

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1')
print ('\n2.- b)')

##Apartado 2 b)
w, num_ite=gd([1,1], 0.1, gradE, E, 1e-14)
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")

##Apartado 2 c)
print('\n2.- c)')
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
x = np.arange(-1, 1, 0.25)
y = np.arange(-1, 1, 0.25)

X, Y = np.meshgrid(x, y)
Z = E([X, Y])
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.view_init(0, 35)
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('Representación de E');
ax.contour3D(X, Y, Z, 50, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


##Apartado 3 a)
# Fijamos la semilla

def f(w): 
    x=w[0]
    y=w[1]
    return (x-2)**2+2*(y+2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
			 
# Derivada parcial de f respecto de x
def fx(w):
    x=w[0]
    y=w[1]
    return 2*(x-2)+4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
# Derivada parcial de f respecto de y
def fy(w):
    x=w[0]
    y=w[1]
    return 4*(y+2)+4*np.pi*np.sin(2*math.pi*x)*np.cos(2*np.pi*y)
#Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, f, max_iters = 50):
    graf=[]
    it=0
    while it<max_iters:
        graf.append(f(w))
        w=w-lr*grad_fun(w)
        it+=1
    
    plt.plot(range(0,max_iters), graf, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()	
    return w

x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(x, y)
Z = f([X, Y])
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

print ('Resultados ejercicio 3.- a)')
print ('\nGrafica con learning rate igual a 0.01')
gd_grafica([1,-1], 0.01, gradf, f)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica([1,-1], 0.1, gradf, f)
input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = 50):
    it=0
    while it<max_iters:
        w=w-lr*grad_fun(w)
        it+=1		
    return w

print ('Punto de inicio: (2.1, -2.1)\n')
w=gd_grafica([2.1,-2.1], 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w=gd_grafica([3.0,-3.0], 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w=gd_grafica([1.5,1.5], 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w=gd_grafica([1.0,-1.0], 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))


