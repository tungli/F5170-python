import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(0,4000,500)
def Fv(v):
    amu = 1.66053904e-27
    m = 40.0*amu
    kB = 1.3806485e-23
    T = 1000.0
    a = m/kB/T
    return np.sqrt(2/np.pi)*v**2*a**(3.0/2.0)*np.exp(-a/2*v**2)

v_mean = 
v_sq_mean = 
v_mp =
v_mean_an = 
v_sq_mean_an = 
v_mp_an =

plt.plot(v,Fv(v))
plt.plot(v_mean,Fv(v_mean),'o')
plt.plot(v_sq_mean,Fv(v_sq_mean),'o')
plt.show(block=True)





