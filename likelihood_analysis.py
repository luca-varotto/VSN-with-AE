import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy.solvers import solve
from sympy import Symbol


############################################################################################################################

from sim_manager import Manager

############################################################################################################################

# 

############################################################################################################################

'''
'''
def create_events_pdf(M, uniform_flag=True):
  if uniform_flag:
    events_pdf = (1/M)*np.ones(M) 
  else:
    events_pdf = np.zeros(M)
    for m in range(M):
      events_pdf[m] = np.random.uniform(0.3,0.8)
    events_pdf /= sum(events_pdf)
  return events_pdf
 
# SETUP PARAMETERS
user_params = {

# UNDERLYING STIMULATION MODEL
'M': 3, # number of events (unknown)
'M_max': 20, # a-priori knowledge on the upper bound on M
'events': None, # events labels [defined later]
'p_events': None, # events a-priori probability (unknwon) [defined later]
'T': None, # underlying stimulation matrix (unknwon) [defined later]

# VSN
'N': 15, # number of nodes

# DATASET
'p_succ': 0.8,  # success probability for each node (success = correct measurement)
'n': int(5*1.0E+3),  # number of measurements
'confidence_range': [0.7,1.0], # confidence associated to a single measurement
'split_ratio': 0.8, # train-test split ratio

# AE
'structure_array': None,  # encoder structure (decoder is symmetric) [defined later]
'epochs': 30, 
'batch_size': 30,
    
# DNN
'use_DNN': True # boolean flag to activate/deactivate the comparison of AE with a DNN
}
user_params['events'] = range(user_params['M'])
user_params['p_events'] = create_events_pdf(user_params['M'],uniform_flag=True)
user_params['structure_array']= [math.ceil(3*user_params['N']/4),math.ceil(user_params['N']/2),math.ceil(user_params['N']/4),2]

# instantiate the simulation manager according to the user-defined parameters
manager = Manager(user_params=user_params)

# define the triggering model through the stimulation matrix (supposed injective)
manager.user_params['T'] = manager.create_T()

T = manager.user_params['T']
p_D = user_params['p_succ']
p_C =0.99
P_C_bar = (1-p_C)/(user_params['M_max']-1)
j =  np.random.randint(low=0,high=user_params['N'])
m = np.random.randint(low=0,high=user_params['M'])
while T[m,j] ==0:
    m = np.random.randint(low=0,high=user_params['M'])
p_h1 = T[m,j]*user_params['p_events'][m]*p_D*p_C +\
    p_D*P_C_bar*(np.sum(T[:,j]*user_params['p_events']) - max(T[m,j]*user_params['p_events'][m],0))
p_h0 = 1 - p_h1
c_min = 0.1

# print('solving')

# x = Symbol('x')
# print(solve( 
#     1.5*(1-x)/(1-c_min)*( (1-c_min**3)/3 -(1-x)*(1+c_min)*(1-c_min**2)/2 +0.25*(1-c_min)*((1+c_min)*(1-x))**2 ) + \
#     1.5*0.25*x*((1-x)*(1+c_min))**2 - 0.5 + 0.25*(1+c_min)*(1-x)
# ))
# print( np.log10(3/4) / np.log10(1-0.3*P_C_bar*p_D) )

fig = plt.figure()
K = np.linspace(1,1.0E+4,1.0E+5)
lamda = np.zeros(len(K))
mu = np.zeros(len(K))
sigma = np.zeros(len(K))
for i in range(len(K)):
    lamda[i] = p_h0**K[i]  
    mu[i]=(1-lamda[i])*(1+c_min)/2
    sigma[i] = ( (1-lamda[i])/(1-c_min)*( \
        (1-c_min**3)/3 - 0.5*(1-lamda[i])*(1+c_min)*(1-c_min**2) +0.25*(((1-lamda[i])*(1+c_min))**2)*(1-c_min) ) + \
        lamda[i]*0.25*((1-lamda[i])*(1+c_min))**2 ) / \
        ( (1-lamda[i])/(1-c_min) - lamda[i] )
plt.subplot(3,1,1)
plt.plot(K,lamda)
plt.subplot(3,1,2)
plt.plot(K,mu)
plt.subplot(3,1,3)
plt.plot(K,sigma)
plt.show()


fig = plt.figure()
K = [5*1.0E+0,1.0E+1,1.0E+4]
for i in range(len(K)):
    lamda = p_h0**K[i]  
    mu=(1-lamda)*(1+c_min)/2
    sigma = ( (1-lamda)/(1-c_min)*( \
        (1-c_min**3)/3 - 0.5*(1-lamda)*(1+c_min)*(1-c_min**2) +0.25*(((1-lamda)*(1+c_min))**2)*(1-c_min) ) + \
        lamda*0.25*((1-lamda)*(1+c_min))**2 ) / \
        ( (1-lamda)/(1-c_min) - lamda )
    t = np.linspace(0,1,1.0E+5)
    p_t = np.zeros(len(t))
    p_t_norm = norm.pdf(t,mu,sigma)
    for j in range(len(t)):
        if t[j] ==0:
            p_t[j] = lamda
        elif t[j] >= c_min:
            p_t[j] = (1- lamda)/(1-c_min)
    markers_on = [0]
    plt.subplot(len(K),1,i+1)
    plt.plot(t,p_t/max(p_t),label=str(K[i]),alpha=0.5, linestyle='--',marker='.',markevery=markers_on)
    plt.plot(t,p_t_norm/max(p_t_norm))
    plt.title('K='+str(K[i]))
plt.show()
