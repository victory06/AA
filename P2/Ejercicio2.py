#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:47:05 2020

@author: victor
"""
import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))



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

#Función que devuelve el signo de f
def signof(x,y,intervalo,a,b):
    return np.sign(y-a*x-b)

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


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

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


print("Apartado a)")

x=[]
for i in range(len(ptos)):
    x.append([1,ptos[i][0],ptos[i][1]])


vini=np.zeros(3,float)
w,it=ajusta_PLA(x,y2,np.inf,np.copy(vini))
delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
F=w[2]*Y+w[1]*X+w[0]
plt.contour(X,Y,F,[0])
plt.show()

for i in range(10):
    w,it=ajusta_PLA(x,y2,np.inf,np.copy(vini))
    print(it)
    print(w)



# Random initializations
iterations = []
for i in range(0,10):
    vini=np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])
    w, it=ajusta_PLA(x,y2,np.inf,vini)
    iterations.append(it)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

y2=setLabels(ptos, signof, 10, 10,[-50,50],a,b)
x=[]
for i in range(len(ptos)):
    x.append([1,ptos[i][0],ptos[i][1]])


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

vini=np.zeros(3,float)
w,it=ajusta_PLA(x,y2,1200,vini)
delta=0.025
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)
F=w[2]*Y+w[1]*X+w[0]
plt.contour(X,Y,F,[0])
plt.show()


print(it)
print(w)

#Sale mala muchas veces porque no tenemos ningún cirterio en el algoritmo
#para ver si una recta es mejor que la anterior, no podemos volver atrás.

# Random initializations
iterations = []
for i in range(0,10):
    vini=np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])
    w, it=ajusta_PLA(x,y2,1200,vini)
    iterations.append(it)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
#Reestablecemos semilla


def sgdRL(x,y,lr,vini):
    w=vini
    epocas=0
    z=np.arange(len(y))
    while True:
        epocas+=1
        w_ant=w
        for n in range(len(x)):
            sum=[]
            aux=-y[z[n]]/(1+np.e**(y[z[n]]*w.dot(x[z[n]])))
            for i in range(len(x[1])):
                sum.append(aux*x[z[n]][i])
            dEin=sum
            w=w-lr*np.asanyarray(dEin)
        if(np.linalg.norm(w_ant - w)<0.01):
            break
        np.random.shuffle(z)
    return w,epocas
#Función Error 
def Err(x,y,w):
    
    sol=0
    for i in range(len(x)):
        sol+=np.log(1+np.exp(-y[i]*w.dot(x[i])))
    return sol/len(y)

#Crear de los datos
ptos=simula_unif(100,2,[0,2])
a, b=simula_recta([0,2])
y2=setLabels(ptos, signof, 101, 101,[0,2],a,b) #101 para que no haya ruido
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


#Aplicamos regresion logística y mostramos datos y gráficas
x=[]
for i in range(len(ptos)):
    x.append([1,ptos[i][0],ptos[i][1]])
vini=np.zeros(3,float)
w,it=sgdRL(x,y2,0.01,vini)
delta=0.025
xrange = np.arange(0, 2, delta)
yrange = np.arange(0, 2, delta)
X, Y = np.meshgrid(xrange,yrange)
F=w[2]*Y+w[1]*X+w[0]
G=Y-a*X-b
plt.contour(X,Y,F,[0],colors=['#101010', '#A0A0A0', '#303030'])
plt.contour(X,Y,G,[0],colors=['#A0A0A0', '#A0A0A0', '#C0C0C0'])
plt.legend(loc=2)
plt.show()
print("Número de iteraciones y vector de pesos:")
print("iteraciones: ",it)
print(w)
print("Error dentro de la muestra: ",Err(x,y2,w))

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


#Apartado 2
print("\nApartado 2")
Errmedio=0

ptos=simula_unif(1000,2,[0,2])
x=[]
for i in range(len(ptos)):
    x.append([1,ptos[i][0],ptos[i][1]])
y2=setLabels(ptos, signof, 101, 101,[0,2],a,b) #101 para que no haya ruido
Errmedio+=Err(x,y2,w)


print("Error fuera de la muestra para una muestra de 1000 puntos",Errmedio)

for j in range(1000):
    ptos=simula_unif(100,2,[0,2])
    x=[]
    for i in range(len(ptos)):
        x.append([1,ptos[i][0],ptos[i][1]])
    y2=setLabels(ptos, signof, 101, 101,[0,2],a,b) #101 para que no haya ruido
    Errmedio+=Err(x,y2,w)

print("Error medio fuera de la muestra en 1000 iteraciones de 100 puntos de muestra aleatorios", Errmedio/1000)





