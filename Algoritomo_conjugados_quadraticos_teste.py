#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importando os arquivos para o codigos
import numpy as np
import sys as ss
import time
x0 = np.array([10,10],dtype=np.float)
A = np.array([(15,0),(0,15)],dtype=np.float)
b = np.array([8,3],dtype=np.float)
c = np.array([0,0])
n = len(b)
x1= np.array([-3., -3.])


# In[11]:


# Calcula o gradiente
def grad(x):
    grad0 = np.dot(A,x) + b
    return grad0
print(grad(x0))


# In[5]:


#Essa é a função
def f(x):
    return (1/2)*(dot(A,x0)) + dot(b,x0) + c


# In[6]:


sigma = np.linalg.eigvals(A)
print(sigma)


# In[7]:


#Essa é direção
d = -grad(x0)


# In[8]:


#Calcula t_k
def t_k1():
    t_k = -1*np.dot(grad(x0).T,d)/(np.dot(np.dot(d.T,A),d))
    return t_k
t_k = t_k1()


# In[9]:


#Calcula x_k
def x_k1():
    x_k= x0 + t_k*d
    return x_k
x_k = x_k1()
print(x_k)


# In[10]:


#Calcula Beta
def Beta_k():    
    Beta  = np.dot(np.dot(d.T,A),grad(x_k))/(np.dot(np.dot(d.T,A),d))
    return Beta
Beta = Beta_k()
print(Beta)


# In[11]:


#Essa é a nova direção
def dk():
    d_k = -grad(x_k) + np.dot(Beta,d)
print(d_k)
d_k = dk()


# In[41]:


def comparar(g,i=0):
    tam = len(g)
    igual = 0
    resposta = " "
    for i in range(len(g)):
        if(g[i] == 0):
            igual = igual + 1
            if(igual == len(g)):
                reposta = "igual"
                break
        else:
            reposta = "Não igual"
            i = i+1
    return reposta


# In[42]:


def conj_grad(x0,A,b):
    k = 0
    g = grad(x0)
    d = -g
    print(g)
    valor_conver = []
    while(k < 101):
        if(comparar(g) == "Não igual"):
            t_k = -1*np.dot(grad(x0).T,d)/(np.dot(np.dot(d.T,A),d))
            x_k =  x0 + np.dot(t_k,d)
            Beta = np.dot(np.dot(d.T,A),grad(x_k))/(np.dot(np.dot(d.T,A),d))
            d_k = -grad(x_k) + np.dot(Beta,d)
            x0 = x_k
            g = grad(x_k)
            print(g)
        elif(k == 100):
            print("Não encontrou ponto de minimo")
            break
        else:
            valor_conver.append(x0)
            print(x0,"Ponto mínimo")
            break
        k += 1
    return valor_conver


# In[43]:


print(conj_grad(x0,A,b))


# In[ ]:





# In[ ]:




