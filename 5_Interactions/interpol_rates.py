import numpy as np
from scipy.interpolate import interp1d

f = open("sigmaion.dat")
d = f.read()
f.close()
data = d.replace('\n',',').split(',')[:-1:]
x = np.array([float(i) for i in data[0::2]])
y = np.array([float(i) for i in data[1::2]])

v = np.linspace(0.0,1e7,int(1e6))
sigma = interp1d(x,y,kind='cubic',bounds_error=False,fill_value=0.0)

def Fv(v):
    m = 9.109e-31 
    kB = 1.3806485e-23
    T = 11604.0*15.0
    a = m/kB/T
    return np.sqrt(2.0/np.pi)*v**2*a**(3.0/2.0)*np.exp(-a/2*v**2)

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma

kr_MB = np.trapz(v*Fv(v)*sigma(v),x=v)
vmean = np.trapz(Fv(v)*v,x=v)
kr_beam = np.trapz(v*gaussian(v,vmean,vmean/100)*sigma(v),x=v)

print(kr_MB,kr_beam)


