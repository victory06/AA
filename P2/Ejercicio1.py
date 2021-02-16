#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:36 2020

@author: victor
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def calcula_fallo(ptos, labl, f):
    neg_mal=0;
    pos_mal=0;
    pos=0;
    neg=0;
    for i in range(len(ptos)):
        if(labl[i]!=f(ptos[i][0],ptos[i][1]) and labl[i]<0):
            neg_mal+=1
            neg+=1
        elif(labl[i]<0):
            neg+=1
        if(labl[i]!=f(ptos[i][0],ptos[i][1]) and labl[i]>0):
            pos_mal+=1
            pos+=1
        elif(labl[i]>0):
            pos+=1
    return pos, neg, pos_mal, neg_mal
            

#Función que devuelve el signo de f
def signof(x,y,intervalo,a,b):
    return np.sign(y-a*x-b)

print ('Ejericio 1 a) Representación de la nube unif\n')
ptos=simula_unif(50,2,[-50,50])
for i in range(50):
    plt.plot(ptos[i][0], ptos[i][1], 'bo')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print ('Ejericio 1 b) Representación de la nube gaus\n')

gaus=simula_gaus(50,2,[5,7])
for i in range(50):
    plt.plot(gaus[i][0], gaus[i][1], 'bo')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print ('Ejercicio 2 a) Establezco etiquetas a los puntos del 1 a)\n')

#Función para establecer las etiquetas
def setLabels(ptos, f, noisepos, noiseneg, interv,a,b):
    y=[]
    indices_pos=[]
    indices_neg=[]
    for i in range(len(ptos)):
        valor=f(ptos[i][0], ptos[i][1], interv,a,b)
        y.append(valor)
        if(valor==-1):
            indices_neg.append(i)
        if(valor==1):
            indices_pos.append(i)
        
    if(noisepos<100):
        num=int((noisepos/100)*len(indices_pos))
        chosen=np.random.randint(0,len(indices_pos)-1, num)
        for i in range(len(chosen)):
            y[indices_pos[chosen[i]]]=y[indices_pos[chosen[i]]]*(-1)
    if(noiseneg<100):
        num=int((noiseneg/100)*len(indices_neg))
        chosen=np.random.randint(0,len(indices_neg)-1, num)
        for i in range(len(chosen)):
            y[indices_neg[chosen[i]]]=y[indices_neg[chosen[i]]]*(-1)
    return y
ptos=simula_unif(100,2,[-50,50])
a, b=simula_recta([-50,50])
y2=setLabels(ptos, signof, 101, 101,[-50,50],a,b) #101 para que no haya ruido



delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=Y-a*X-b
plt.contour(X,Y,G,[0])

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

print ('Ejercicio 2 b) Establezco ruido\n')
#10% de ruido a los puntos anteriores con la recta anterior

y2=setLabels(ptos, signof, 10, 10,[-50,50],a,b) 

delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=Y-a*X-b
plt.contour(X,Y,G,[0])


#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()
####################################33
neg_mal=0;
pos_mal=0;
pos=0;
neg=0;
for i in range(len(ptos)):
    if(y2[i]<0):
        if(y2[i]!=signof(ptos[i][0],ptos[i][1],[-50,50],a,b)):
            pos_mal+=1
        neg+=1
    if(y2[i]>0):
        if(y2[i]!=signof(ptos[i][0],ptos[i][1],[-50,50],a,b)):
            neg_mal+=1
        pos+=1

##########################################3
print(neg_mal)
print(neg)
print(pos_mal)
print(pos)
print("ERROR en la recta de las negativas:", neg_mal*100/neg)
print("ERROR en la recta de las positivas:", pos_mal*100/pos)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Ejercicio 2c). Uso diferentes funciones para las etiquetas:\n')

def f1(x,y):
    return np.sign((x-10)**2+(y-20)**2-400)

def f2(x,y):
    return np.sign(0.5*(x+10)**2+(y-20)**2-400)

def f3(x,y):
    return np.sign(0.5*(x-10)**2+(y+20)**2-400)

def f4(x,y):
    return np.sign(y-20*x**2-5*x+3)

#Función para establecer las etiquetas adaptado a este ejercicio
def setLabels3(ptos, f, noisepos, noiseneg, interv):
    y=[]
    indices_pos=[]
    indices_neg=[]
    for i in range(len(ptos)):
        valor=f(ptos[i][0], ptos[i][1], interv)
        y.append(valor)
        if(valor==-1):
            indices_neg.append(i)
        if(valor==1):
            indices_pos.append(i)
        
    if(noisepos<100):
        num=int((noisepos/100)*len(indices_pos))
        chosen=np.random.randint(0,len(indices_pos)-1, num)
        for i in range(len(chosen)):
            y[indices_pos[chosen[i]]]=y[indices_pos[chosen[i]]]*(-1)
    if(noisepos<100):
        num=int((noisepos/100)*len(indices_neg))
        chosen=np.random.randint(0,len(indices_neg)-1, num)
        for i in range(len(chosen)):
            y[indices_neg[chosen[i]]]=y[indices_neg[chosen[i]]]*(-1)
    return y

print ('Para la función 1:\n')



delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=f1(X,Y)
plt.contour(X,Y,G,[0])

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()

pos, neg, pos_mal, neg_mal=calcula_fallo(ptos,y2,f1)
print("ERROR en la funcion de las negativas:", neg_mal*100/neg)
print("ERROR en la funcion de las positivas:", pos_mal*100/pos)


input("\n--- Pulsar tecla para continuar ---\n")

print ('Para la función 2:\n')




delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=f2(X,Y)
plt.contour(X,Y,G,[0])

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()

pos, neg, pos_mal, neg_mal=calcula_fallo(ptos,y2,f2)
print("ERROR en la funcion de las negativas:", neg_mal*100/neg)
print("ERROR en la funcion de las positivas:", pos_mal*100/pos)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Para la función 3:\n')



delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=f3(X,Y)
plt.contour(X,Y,G,[0])

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()

pos, neg, pos_mal, neg_mal=calcula_fallo(ptos,y2,f3)
print("ERROR en la funcion de las negativas:", neg_mal*100/neg)
print("ERROR en la funcion de las positivas:", pos_mal*100/pos)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Para la función 4:\n')


delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
G=f4(X,Y)
plt.contour(X,Y,G,[0])

#Pintamos resultados:
etiqueta1=0
etiqueta2=0
for i in range(len(ptos)):
    if(y2[i]==-1):
        if(etiqueta1==0):
                plt.plot(ptos[i][0], ptos[i][1], 'bo', label='-1')
                etiqueta1=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'bo')
    if(y2[i]==1):
        if(etiqueta2==0):
            plt.plot(ptos[i][0], ptos[i][1],  'ro', label='1')
            etiqueta2=1
        else:
            plt.plot(ptos[i][0], ptos[i][1],  'ro')
plt.legend(loc=2)
plt.show()

pos, neg, pos_mal, neg_mal=calcula_fallo(ptos,y2,f4)
print("ERROR en la funcion de las negativas:", neg_mal*100/neg)
print("ERROR en la funcion de las positivas:", pos_mal*100/pos)

