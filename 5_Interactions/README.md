# Particle interactions in plasma

This chapter is mostly about numerical integration and interpolation applied to the study of distribution function.

## Numerical integration
We will be using the trapezoidal rule here.
This is not the best choice but (hopefully) it will be enough.
If you are interested in better algorithms read about the [Newton-Cotes rules](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas) or the [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature)

To numerically integrate a function, you could just sum the functional values evaluated in the center of some intervals multiplied by the width of the interval.
This corresponds to dividing the function into rectangles.
This is called the midpoint rule. This is not optimal, because you would need a large number of intervals and, therefore, function evaluations to get a good accuracy.

The trapeziodal rule is a simple improvement.
Instead of dividing into rectangles use trapezoids.

In the NumPy package, there already is a implemented function that uses this method to integrate an array of functional values called `trapz()`.
Take a look at the [reference page](https://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html).

Here is a simple example:
```python
import numpy as np

def f(x):
    return 1/(1 + x**2)

for i in [100,1000,10000,100000]:
    x = np.linspace(0,1,i)
    y = f(x)
    integral = np.trapz(y,x=x)
    print(4*integral)

print(np.pi)
```

## Maxwell-Boltzmann distribution
Now, you should apply numerical integration to Maxwell-Boltzmann distribution to calculate mean values and compare these with the theoretical values.
Here is a [script](https://github.com/tungli/F5170-python/blob/master/5_Interactions/maxwell.py) to guide you.
```python
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

v_mean_an = 
v_sq_mean_an =
v_mp_an =

plt.plot(v,Fv(v))
plt.plot(v_mean,Fv(v_mean),'o')
plt.plot(v_sq_mean,Fv(v_sq_mean),'o')
plt.show(block=True)
```

## Time evolution of distribution function
If all goes well, this [script](https://github.com/tungli/F5170-python/blob/master/5_Interactions/evol.py) play an animation showing a simple time evolution of simple distribution function.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

v = np.linspace(-2000,2000,200)
t = np.linspace(0,1e-8,100)
amu = 1.66053904e-27
m = 40.0*amu
kB = 1.3806485e-23
T = 1000.0
coll_freq = 5e8
```
This is just the usual importing and defining constants.

```python
def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma

fx0 = gaussian(v,500.0,10.0)
fy0 = np.zeros(np.shape(v))
fz0 = np.zeros(np.shape(v))

v_sq = np.sqrt(np.trapz((fx0+fy0+fz0)*v**2,x=v))
Teq = m*v_sq**2/kB/3

fx_eq = np.sqrt(m/(2*np.pi*kB*Teq))*np.exp(-m*v**2/(2*kB*Teq))
fy_eq = np.sqrt(m/(2*np.pi*kB*Teq))*np.exp(-m*v**2/(2*kB*Teq))
fz_eq = np.sqrt(m/(2*np.pi*kB*Teq))*np.exp(-m*v**2/(2*kB*Teq))

def calc_distribution(feq,f0,t,coll_f):
    return feq + (f0 - feq)*np.exp(-t*coll_f) 
```
Here we define the initial and equilibrium distribution functions and a function which calculates the distribution function for any time.

```python
def update_line(num, data, l):
    l[0].set_data(v,data[num][0])
    l[1].set_data(v,data[num][1])
    l[2].set_data(v,data[num][2])
    return l

fig = plt.figure(figsize=(10,7))

data = [[calc_distribution(fx_eq,fx0,i,coll_freq),
        calc_distribution(fy_eq,fy0,i,coll_freq),calc_distribution(fz_eq,fz0,i,coll_freq)]
        for i in t]

ax1 = fig.add_subplot(1,3,1)
l1, = ax1.plot([], [], lw=2, color='r')
ax1.set_xlim(-2000, 2000)
ax1.set_ylim(0,np.max(fx0))
#Here you can set up the labels, title, etc.

ax2 = fig.add_subplot(1,3,2)
l2, = ax2.plot([], [], lw=2, color='r')
ax2.set_xlim(-2000, 2000)
ax2.set_ylim(0,np.max(fx0))
#Here you can set up the labels, title, etc.


ax3 = fig.add_subplot(1,3,3)
l3, = ax3.plot([], [], lw=2, color='r')
ax3.set_xlim(-2000, 2000)
ax3.set_ylim(0,np.max(fx0))
#Here you can set up the labels, title, etc.

l = [l1,l2,l3]

line_ani = animation.FuncAnimation(fig, update_line, 100, fargs=(data, l),
                                   interval=50, blit=True)
plt.show()
```
These just the things for the animation.
You can add some labels or change the colors, etc.

Here is an example of one frame from the animation:
![Evolution](https://github.com/tungli/F5170-python/blob/master/5_Interactions/evol_ex.svg)

## Rate coefficients
You will find the data you need in this directory, here are the [rate coefficents](https://github.com/tungli/F5170-python/blob/master/5_Interactions/sigmaion.dat) to interpolate.

We will be using the `interp1d()` from the `scipy.interpolate` subpackage.
Take a look at the [online reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).

```python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

f = open("sigmaion.dat")
d = f.read()
f.close()
data = d.replace('\n',',').split(',')[:-1:]
x = np.array([float(i) for i in data[0::2]])
y = np.array([float(i) for i in data[1::2]])

v = np.linspace(0.0,1e7,1e6)
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
```
This is a [script](https://github.com/tungli/F5170-python/blob/master/5_Interactions/interpol_rates.py) you can use for the exercise.
In involves importing some packages, getting the data from a file, interpolating and integrating some functions.

The interpolation takes place in this line:
```python
sigma = interp1d(x,y,kind='cubic',bounds_error=False,fill_value=0.0)
```
Notice the arguments - `'cubic'` means to use the cubic interpolation, `bounds_error=False` is to allow extrapolation as well, nevertheless, all the values that are extrapolated will be zero, since `fill_value=0.0`.
The `sigma` object will act like a function - meaning you can write `sigma(0.5e7)` to get the value at `0.5e7`.

Here is plot with the interpolation:
![Rates Interpolation](https://github.com/tungli/F5170-python/blob/master/5_Interactions/rates_interp.svg)


