import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

p = 1e5
kB = 1.38e-23
Tg = 400.0
Te = 2.0

tsteps = 1000
tspan = np.logspace(-11,-6,tsteps)

nArs_init = 1e12
nArp_init = 1e12
nAr2p_init = 1e12
ne_init = nArp_init +  nAr2p_init
initial = [nArs_init, nArp_init, nAr2p_init, ne_init]

def odefun(y,t):
    nArs = y[0]
    nArp = y[1]
    nAr2p = y[2]
    ne = y[3]

    nAr = p/(kB*Tg) - nArs - 0.5*nAr2p - nArp
    k1 = f_k1(Te,Tg)    
    k2 = f_k2(Te,Tg)
    k3 = f_k3(Te,Tg)
    k4 = f_k4(Te,Tg)
    k5 = f_k5(Te,Tg)
    k6 = f_k6(Te,Tg)
    k7 = f_k7(Te,Tg)
    k8 = f_k8(Te,Tg)
    k9 = f_k9(Te,Tg)
    k10 = f_k10(Te,Tg)
    k11 = f_k11(Te,Tg)

    SArs = k1*ne*nAr\
            -k2*ne*nArs\
            -k4*ne*nArs\
            +k6*ne*nAr2p\
            -2*k10*nArs**2\
            -k11*nArs*nAr
    #You need to fill in the correct expressions
    SArp = 1.0
    SAr2p = 1.0
    Se = 1.0

    return np.array([SArs,SArp,SAr2p,Se])

#You need to define these functions
def f_k1(Te, Tg):
    return 1.0
def f_k2(Te, Tg):
    return 1.0
def f_k3(Te, Tg):
    return 1.0
def f_k4(Te, Tg):
    return 1.0
def f_k5(Te, Tg):
    return 1.0
def f_k6(Te, Tg):
    return 1.0
def f_k7(Te, Tg):
    return 1.0
def f_k8(Te, Tg):
    return 1.0
def f_k9(Te, Tg):
    return 1.0
def f_k10(Te, Tg):
    return 1.0
def f_k11(Te, Tg):
    return 1.0

y = odeint(odefun,initial,tspan) 

nAr = p/(kB*Tg) - y[:,0]- 0.5*y[:,2]- y[:,1]

plt.loglog(tspan,nAr,'c',basex=10, label='Ar')
plt.loglog(tspan,y[:,0],'r',basex=10, label='Ar^*')
plt.loglog(tspan,y[:,1],'b',basex=10, label='Ar^+')
plt.loglog(tspan,y[:,2],'m',basex=10, label='Ar_2^+')
plt.loglog(tspan,y[:,3],'k',basex=10, label='el.')
plt.show()
