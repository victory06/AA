#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:57:28 2020

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Fijamos la semilla
np.random.seed(1)

###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

# Funcion para calcular el error
def Err(x,y,w):
    
    sol=0
    for i in range(len(x)):
        sol+=(w.dot(x[i])-y[i])**2
    return sol/len(y)

# Funcion para calcular la derivada del error error
def Errd(x,y,w,j,tam_minibatch):
    ls=[]
    for i in range(len(y)):
        ls.append(i)
    yi=random.sample(ls,tam_minibatch) #No se repiten numeros
    x2=[]
    y2=[]
    for i in range(len(yi)):
        x2.append(x[yi[i]])
        y2.append(y[yi[i]])
    sol=0
    for i in range(tam_minibatch):
        sol+=x2[i][j]*(w.dot(x2[i])-y2[i])
    return (2*sol)/tam_minibatch
	
# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
    w=np.zeros(len(x[0]))
    it=0
    while it<max_iters:
        w_ant=np.copy(w)
        for j in range(len(w)):
            w[j]=w[j]-lr*Errd(x,y,w_ant,j,tam_minibatch)
        it+=1
    return w

#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()

vini=np.zeros(3,float)
w=sgd(x,y,0.01,500,32)
delta=0.025
xrange = np.arange(0, 1, delta)
yrange = np.arange(-7, 0, delta)
X, Y = np.meshgrid(xrange,yrange)
F=w[2]*Y+w[1]*X+w[0]
plt.contour(X,Y,F,[0],colors=['#101010', '#A0A0A0', '#303030'])
plt.legend(loc=2)
plt.show()

print(w)
print("Error en el training: ", Err(x,y,w))
print("Error en el test: ", Err(x_test,y_test,w))

input("\n--- Pulsar tecla para continuar ---\n")

print("Esperar unos segundos")
  
def ErrPocket(x,y,w):
    w=np.asanyarray(w)
    err=0
    for i in range(len(x)):
        if(np.sign(w.dot(x[i]))!=y[i]):
            err+=1
    return err/len(x)

def ajusta_PLA(datos, label, max_iter, vini):
    w=vini
    cambio=1
    it=0
    while cambio and it<max_iter:
        cambio=0
        it+=1
        for i in range(len(datos)):
            if(np.sign(w.dot(datos[i]))!=label[i]):
                for j in range(len(datos[i])):
                    w[j]=w[j]+label[i]*datos[i][j]
                cambio=1
    return w, it

def pocket(x,y,w0,max_iters):
    best_w=np.copy(w0)
    best_err=ErrPocket(x,y,w0)
    w_actual=w0
    err_actual=0
    for i in range(max_iters):
        w_actual,it=ajusta_PLA(x,y,100,best_w)
        err_actual=ErrPocket(x,y,w_actual)
        if(err_actual<best_err):
            best_w=np.copy(w_actual)
            best_err=err_actual
    return best_w, best_err



#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()

vini=np.zeros(3,float)
w,err=pocket(x,y,np.copy(w),20)
delta=0.025
xrange = np.arange(0, 1, delta)
yrange = np.arange(-7, 0, delta)
X, Y = np.meshgrid(xrange,yrange)
F=w[2]*Y+w[1]*X+w[0]
plt.contour(X,Y,F,[0],colors=['#101010', '#A0A0A0', '#303030'])
plt.legend(loc=2)
plt.show()

print(w)
print("Error en la muestra: ",err)
print("Error fuera de la muestra (test): ", ErrPocket(x_test,y_test,w))
    



input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

print("Cota para el Err out basado en el in: ", err+np.sqrt(1/(2*len(x))*np.log(2/0.05)))
print("Cota para el Err out basado en el test: ", ErrPocket(x_test,y_test,w)+np.sqrt(1/(2*len(x))*np.log(2/0.05)))

