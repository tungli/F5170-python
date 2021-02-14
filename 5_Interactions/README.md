# Particle interactions in plasma

This chapter is mostly about numerical integration and interpolation applied to the study of distribution functions.

## Numerical integration
We will be using the trapezoidal rule here.
This is not the best choice but (hopefully) it will be enough.
If you are interested in better algorithms read about the [Newton-Cotes rules](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas) or the [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature)

To numerically integrate a function, you could just sum the functional values evaluated in the center of some intervals multiplied by the width of the interval.
This corresponds to dividing the function into rectangles.
This is called the midpoint rule. This is not optimal, because you would need a large number of intervals and, therefore, function evaluations to get a good accuracy.

The trapezoidal rule is a simple improvement.
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

## Exercises
>  **Exercise 1**
>  * Complete this [script](https://github.com/tungli/F5170-python/blob/master/5_Interactions/maxwell.py). The script should plot the Maxwell-Boltzmann distribution with respect the velocity magnitude and a few points of interest, namely:
>    1. The mean speed
>    2. The mean squared speed (norm of the velocity vector)
>    3. The most probable speed
>  * Calculate these values analytically and numerically. Compare them and explain any differences.
>  * Plot the distributions and the values of distributions at the calculated speeds. Which of the 3 speeds is the lowest. Does their order change with temperature?
>  * Change the number of points of integration. How does it affect the result of the integration?
>  
>  **Exercise 2**
>  * The mass of nitrogen molecules is 28 a.m.u. and their number density at ambient conditions is approximately *1.7e25* m<sup>−3</sup>. How many nitrogen molecules in your room are faster than 50, 500, 1000, 2500, 5000 and 10000 m/s?
>  
>  **Exercise 3**
>  The figure below shows three Maxwell-Boltzmann distributions.
>  * Assume that the distributions differ only in temperature. Which of the distributions has the highest temperature and which the lowest?
>  * Assume that the distributions differ only in particle mass. Which of the distributions has the highest mass and which the lowest?
>  
>  ![dists](https://github.com/tungli/F5170-python/blob/master/5_Interactions/dists.svg)
>  

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
v_mp =
v_mean_an = 
v_sq_mean_an =
v_mp_an =

plt.plot(v,Fv(v))
plt.plot(v_mean,Fv(v_mean),'o')
plt.plot(v_sq_mean,Fv(v_sq_mean),'o')
plt.show(block=True)
```

## Time evolution of a distribution function
In the previous section you analyzed the distribution function of velocity magnitude.
In this section you will be using 3 distribution functions for 3 Cartesian components of the velocity vector and you will look at a simple time evolution of a system described by distribution functions.

Our simple model is based on the [Boltzmann transport equation](https://en.wikipedia.org/wiki/Boltzmann_equation)

![boltzmannEq](http://mathurl.com/y9qsxtzt.png)

We will assume homogeneity in spatial coordinates, zero external force and for the collision term we will assume the following form ([Krook](https://en.wikipedia.org/wiki/Bhatnagar%E2%80%93Gross%E2%80%93Krook_operator)):

![krook](http://mathurl.com/y77eakb6.png)

where *ν<sub>m</sub>* is the collision frequency.

Our kinetic equation therefore simplifies greatly, in fact, it can be integrated analytically:

![kinetic](http://mathurl.com/ya2qhqxq.png)


## Exercises
>  **Exercise 4**
>  If all goes well, this [script](https://github.com/tungli/F5170-python/blob/master/5_Interactions/evol.py) play an animation showing a simple time evolution of simple distribution function.
>  Answer the following questions:
>  * What is the physical meaning of the initial condition for the distribution function?
>  * What kind of particles could the distribution functions describe?
>  * The collision frequency is *5e8* Hz, which is a reasonable value. What is the time necessary for reaching the equilibrium?
>  * Try increasing and decreasing the collision frequency in the script. What happens with the time necessary for reaching equilibrium and why?
  

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
If you have a system described by a distribution function you have access to macroscopic properties of the system.
Here, we will demonstrate this by calculating the rate constants of reactions taking place in plasma.
Rate constants are calculated from collisional cross sections *σ* of a reaction indexed *r* as:

![rateintegral](http://mathurl.com/yde22kcc.png)

Cross sections of most reactions do not have functional representations - they are available only as tabulated data.

You will find the data you need in this directory, here are the [cross sections](https://github.com/tungli/F5170-python/blob/master/5_Interactions/sigmaion.dat) to interpolate.
The first column of the data are speeds in m/s, the second column are the cross sections in m<sup>2</sup>.
The cross section are those of argon ionization by electron impact:

![arelimpioni](http://mathurl.com/ybcgs3t8.png)


>  ## Exercises
>  **Exercise 5**
>  * Run this [scipt](https://github.com/tungli/F5170-python/blob/master/5_Interactions/interpol_rates.py). It calculates the rate constant from the cross section data using two different distribution function - a Maxwell-Boltzmann and a nearly mono-energetic distribution (delta function approximated by a Gaussian function). You will notice that the mean velocity is the same for both distributions but the rate constants differ. Provide an explanation.
>  
>  **Exercise 6**
>  * Modify the script so that it plots cross section as function of speed.
>  * Answer the following questions:
>    1. What is the ionization threshold in electronvolts?
>    2. Where does the cross section reach the maximum value?
>  
>  **Exercise 7**
>  * Run the script for several electron temperatures.
>  * What happens with the rate coefficients with increasing electron temperature?
>  * Is there electron temperature for which the rate coefficient for the nearly mono-energetic beam exceeds the Maxwell-Boltzmann coefficient? Provide an explanation why this is/is not possible
>  
>  **Advanced Exercise**
>  * Rewrite the scipt so that it uses electron energy rather than electron speed. Using electron energy is more common in plasma physics.
  

We will be using the `interp1d()` from the `scipy.interpolate` sub-package.
Take a look at the [online reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).

```python
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
```
The scipt involves importing some packages, getting the data from a file, interpolating and integrating some functions.

The interpolation takes place in this line:
```python
sigma = interp1d(x,y,kind='cubic',bounds_error=False,fill_value=0.0)
```
Notice the arguments - `'cubic'` means to use the cubic interpolation, `bounds_error=False` is to allow extrapolation as well, nevertheless, all the values that are extrapolated will be zero, since `fill_value=0.0`.
The `sigma` object will act like a function - meaning you can write `sigma(0.5e7)` to get the value at `0.5e7`.

Here is an example plot with the interpolation:

![Rates Interpolation](https://github.com/tungli/F5170-python/blob/master/5_Interactions/rates_interp.svg)


