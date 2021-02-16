#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:26:07 2020

@author: victor
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from mpl_toolkits import mplot3d
# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################



#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y
	
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
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
    w=np.zeros(len(x[0]))
    w=np.linalg.pinv(x).dot(y)
    return w

# Pintamos los datos y las soluciones
def RepresentaDatos(x,y, w, x0, y0, labelx, labely):
    etiqueta1=0
    etiqueta2=0
    for i in range(len(x)):
        if(y[i]==-1):
            if(etiqueta1==0):
                plt.plot(x[i][1], x[i][2], 'bo', label='1')
                etiqueta1=1
            else:
                plt.plot(x[i][1], x[i][2], 'bo')
        if(y[i]==1):
            if(etiqueta2==0):
                plt.plot(x[i][1], x[i][2], 'ro', label='5')
                etiqueta2=1
            else:
                plt.plot(x[i][1], x[i][2], 'ro')
    sols_x=[x0,y0]
    sols_y=[w[0]/-w[2]+(w[1]/-w[2])*x0, w[0]/-w[2]+(w[1]/-w[2])*y0]
    plt.plot(sols_x, sols_y, linewidth=2.0, label='recta regresión')
    plt.legend(loc=2,)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()	



# Lectura de los datos de entrenamiento
x, y = readData("./datos/X_train.npy", "./datos/y_train.npy")
# Lectura de los datos para el test
x_test, y_test = readData("./datos/X_test.npy", "./datos/y_test.npy")


print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x, y, 0.01, 100, 32)

print ('Representación Gráfica de datos y soluciones para el train:\n')
RepresentaDatos(x,y,w, 0.0,0.6, 'Simetría', 'Intensidad')

print ('Representación Gráfica de datos y soluciones para el test:\n')
RepresentaDatos(x_test,y_test,w, 0.0,0.6,'Simetría', 'Intensidad')

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test,y_test,w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('Representación Gráfica de datos y soluciones para el train:\n')
RepresentaDatos(x,y,w, 0.0,0.6,'Simetría', 'Intensidad')

print ('Representación Gráfica de datos y soluciones para el test:\n')
RepresentaDatos(x_test,y_test,w, 0.0,0.6,'Simetría', 'Intensidad')

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))



input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    ptosy=np.random.uniform(-size,size,N)
    puntosy=[]
    puntosx=[]
    for i in range(N):
        puntosx.append(i*2*size/N-size)
        puntosy.append(ptosy[i])
    return [puntosx,puntosy]
ptos=simula_unif(1000, 2, 1)



# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2 a)\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

plt.plot(ptos[0], ptos[1], 'bo')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")
print ('Ejercicio 2 b) \n')
print('Representacion de los puntos con sus etiquetas: \n')
def f(x1,x2):
    return np.sign((x1-0.2)**2+x2**2-0.6)

#Función para establecer las etiquetas
def setLabels(ptos, f, noise):
    y=[]
    for i in range(len(ptos[0])):
        y.append(f(ptos[0][i], ptos[1][i]))
    if(noise<100):
        num=int((noise/100)*len(ptos[0]))
        chosen=np.random.randint(0,len(ptos[0])-1, num)
    for i in range(len(chosen)):
        y[chosen[i]]=y[chosen[i]]*(-1)
    return y

y2=setLabels(ptos, f, 10)

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos[0])):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[0][i], ptos[1][i], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[0][i], ptos[1][i],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[0][i], ptos[1][i],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[0][i], ptos[1][i],  'ro')
plt.show()

print ('Ejercicio 2 c) \n')

ptos=np.transpose(ptos)
ptosc=[]
#Creamos array de los puntos con un 1 al principio, como indica el ejercicio
for i in range(len(y2)):
    ptosc.append(np.array([1,ptos[i][0], ptos[i][1]]))
#Gradiente:
z= sgd(ptosc, y2, 0.1, 100, 32)
x0=(z[0]/-z[1])+(z[2]/-z[1])*(-1)
y0=(z[0]/-z[1])+(z[2]/-z[1])*(1)
print('Representacion del gradiente: \n')
RepresentaDatos(ptosc, y2, z, x0, y0, 'x','y')
print('Error in del gradiente descendiente estocastico: \n')
print(Err(ptosc, y2, z))
input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ('Ejercicio 2 d) (Esperar unos segundos)\n')
Ein_media=0
Eout_media=0

#Repetimos mil veces
for i in range(1000):
    ptos=simula_unif(1000, 2, 1)
    y2=setLabels(ptos, f, 10)
    ptos=np.transpose(ptos)
    ptosc=[]
    for i in range(len(y2)):
       ptosc.append(np.array([1,ptos[i][0], ptos[i][1]]))
    z= sgd(ptosc, y2, 0.1, 100, 32)
    Ein_media+=Err(ptosc, y2, z)
    
    #Puntos para calcular Eout
    ptoso=simula_unif(1000, 2, 1)
    y2o=setLabels(ptoso, f, 10)
    ptoso=np.transpose(ptoso)
    ptosco=[]
    for i in range(len(y2o)):
       ptosco.append(np.array([1,ptoso[i][0], ptoso[i][1]]))
    Eout_media+=Err(ptosco, y2o, z)
    

Ein_media=Ein_media/1000
Eout_media=Eout_media/1000
 
print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")



print ('Ejercicio 2 apartados finales: \n')

ptos=simula_unif(1000, 2, 1)
y2=setLabels(ptos, f, 10)
ptos=np.transpose(ptos)
ptosc=[]
for i in range(len(y2)):
    ptosc.append(np.array([1,ptos[i][0], ptos[i][1], ptos[i][0]*ptos[i][1], ptos[i][0]**2,ptos[i][1]**2]))
z= sgd(ptosc, y2, 0.1, 100, 32)

#Pintamos la elipse resultante y los puntos:
delta=0.025
xrange = np.arange(-1, 1, delta)
yrange = np.arange(-1, 1, delta)
X, Y = np.meshgrid(xrange,yrange)
G=z[0]+X*z[1]+Y*z[2]+X*Y*z[3]+(X**2)*z[4]+(Y**2)*z[5]
plt.contour(X,Y,G,[0])
etiqueta1=0
etiqueta2=0
for i in range(len(ptosc)):
    if(y2[i]==-1):
        if(etiqueta1==0):
            plt.plot(ptosc[i][1], ptosc[i][2], 'bo', label='1')
            etiqueta1=1
        else:
            plt.plot(ptosc[i][1], ptosc[i][2], 'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptosc[i][1], ptosc[i][2], 'ro', label='5')
            etiqueta2=1
        else:
            plt.plot(ptosc[i][1], ptosc[i][2], 'ro')
plt.show()

print('Error in del gradiente descendiente estocastico con la segunda característica: \n')
print(Err(ptosc, y2, z))

input("\n--- Pulsar tecla para continuar ---\n")
print ('Ejercicio 2 d) con la segunda característica (Esperar unos segundos)\n')
Ein_media=0
Eout_media=0

#Repetimos mil veces
for i in range(1000):
    ptos=simula_unif(1000, 2, 1)
    y2=setLabels(ptos, f, 10)
    ptos=np.transpose(ptos)
    ptosc=[]
    for i in range(len(y2)):
        ptosc.append(np.array([1,ptos[i][0], ptos[i][1], ptos[i][0]*ptos[i][1], ptos[i][0]**2,ptos[i][1]**2]))
    z= sgd(ptosc, y2, 0.1, 100, 32)
    Ein_media+=Err(ptosc, y2, z)
    
    #Puntos para calcular Eout
    ptoso=simula_unif(1000, 2, 1)
    y2o=setLabels(ptoso, f, 10)
    ptoso=np.transpose(ptoso)
    ptosco=[]
    for i in range(len(y2)):
        ptosco.append(np.array([1,ptoso[i][0], ptoso[i][1], ptoso[i][0]*ptoso[i][1], ptoso[i][0]**2,ptoso[i][1]**2]))
    Eout_media+=Err(ptosco, y2o, z)
    

Ein_media=Ein_media/1000
Eout_media=Eout_media/1000
 
print ('Errores Ein y Eout medios con la segunda característica tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)


#------------------------------------------------------------





