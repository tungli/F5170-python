# Particle balance in plasma

![Concentration Evolution](https://github.com/tungli/F5170-python/blob/master/6_Balance/reactions.svg)

This chapter is about reactions in plasma.
Plasmas usually contain a large number of species.
These can be electrons, different elements or molecules, as well as ionized and excited species.
It can be very useful be able to predict the concentration of species in time and space.

Here, we will restrict our description to time evolution and assume two different temperatures - one for electrons, one for the heavy particles.
The reaction scheme we will use is one for argon plasma at atmospheric pressure.
In the picture below you will find the reactions and reaction rate temperature dependencies.

![scheme](https://github.com/tungli/F5170-python/blob/master/6_Balance/rates_table.png)

To transform a set of chemical reactions to a set of ordinary differential equations (ODEs) describing the evolution of every species here is a general approach:
 * Take the *r*th equation of *M* reactions in total It has the form of [reac](http://mathurl.com/ycnjqt5p.png) where *a* and *b* are the stoichometric coefficients (always nonnegative) of species *X*. Generally, there are two rate constants - forward and backward rates.
 * The ODE for the *i*th specie is:

![masterODE](http://mathurl.com/ycnjqt5p.png)

we used `[]` to represent the concentration.

This may look scary, so here is an example with a one-way reaction:

![ex](http://mathurl.com/ydccwmfy.png)


## Implementation
The whole [script](https://github.com/tungli/F5170-python/blob/master/6_Balance/odesolve.py) is in this repository.
The details are discussed below.
```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt 
p = 1e5
kB = 1.38e-23
Tg = 400.0
Te = 2.0
```
We import NumPy, the ODE solver and a plotting package.
Then we define some constants.

```python
tsteps = 1000
tspan = np.logspace(-11,-6,tsteps)

nArs_init = 1e12
nArp_init = 1e12
nAr2p_init = 1e12
ne_init = nArp_init +  nAr2p_init
initial = [nArs_init, nArp_init, nAr2p_init, ne_init]
```
To solve the system of ODEs, we need to define the time span of the integration and the initial conditions.

```python
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
            +k11*nArs*nAr
    #You need to fill in the correct expressions
    SArp = 1.0
    SAr2p = 1.0
    Se = 1.0

    return np.array([SArs,SArp,SAr2p,Se])
```
This is the first derivatives function.
The functions for the rate constants are missing, you will need to implement them, as well as the right-hand sides for the derivatives.

```python
y = odeint(odefun,initial,tspan) 
```
We use the `odeint()` function to solve the problem numerically.

```python
nAr = p/(kB*Tg) - y[:,0]- 0.5*y[:,2]- y[:,1]

plt.loglog(tspan,nAr,'c',basex=10, label='Ar')
plt.loglog(tspan,y[:,0],'r',basex=10, label='Ar^*')
plt.loglog(tspan,y[:,1],'b',basex=10, label='Ar^+')
plt.loglog(tspan,y[:,2],'m',basex=10, label='Ar_2^+')
plt.loglog(tspan,y[:,3],'k',basex=10, label='el.')
plt.show()
```
Finally, we plot our results using log-log plots.



