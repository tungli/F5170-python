# Particle drift
   
![Drift](https://github.com/tungli/F5170-python/blob/master/3_Motion/drift.svg)
## Mathematical formulation

The exercises want us to solve the equation of motion for a charged particle in a electromagnetic field.
The equation of motion is a set of 3 (x,y,z) ordinary differential equations (ODE) of second order.
To solve a system of ODEs, we usually transform the set of higher order equations to first order equations.
We will create more variables by doing this but that's fine.
We express the first order derivatives as functions of variables and other first order derivatives.
We can then create a function in Python which for a given set of values of variables evaluates the derivatives.
This is the function that a general ODE solver needs.

In our problem we treat positions and velocities as variables and express their derivatives as functions of these variables:

![Equation of Motion](http://mathurl.com/yaan82k2.png)

where *F* is the Lorentz force.

**Note:** There is another way! Here a general approach to ODE is shown but to integrate second order ODEs it is usually not necessary to transform to first order set. 
Another way is, for example, to use the *leapfrog* algorithm. 
If you are not comfortable with using a black-box solver, this is the way for you. Try googling "Boris leapfrog" and implementing it in your code.

## Exercises

>  **Exercise 1**
>  * Run the [script](https://github.com/tungli/F5170-python/blob/master/3_Motion/motion.py).  
>  * Configure the velocity, position and fields as you want.  
>  * Configure the mass and charge to that of an electron.  
>  * What kind of drift do you observe?  
>  * What is the direction of the drift for an electron a what for a positron?  
>  
>  **Exercise 2**
>  * Now configure the parameters for a proton
>  * How many times do you have to increase/decrease the time scale for the plot of the trajectory to be comparable to that of an electron
>  * Compare the amplitudes of the oscillation and the magnitudes of the drift velocities for proton and electron
>  
>  **Exercise 3**
>  * Study a charged particle in the following field and with the following velocity:
>  
> ![prob3](http://mathurl.com/ycp4a5wj.png)
>  
>  * Do you observe any drift? If yes, what parameters did you use? What is the direction of the drift velocity for an electron and a positron?
>  * Try to match your observations with theoretical predictions.
>  
>  **Advanced exercise**
>  * Study both an electron and a proton in an electric field which varies harmonically with time and in uniform magnetic field. Try different frequencies.
>  * How do they react to the field?
>  * Compare the effect for various frequencies 

## Implementation
In this implementation we will be using the `odeint` solver from the `scipy.integrate` sub-package which is a Python wrap around the *ODEPACK* library written in Fortran.
To understand the inputs, outputs and options of the solver, you can check out the [online reference](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.odeint.html).

You can take a look at a basic script that integrates the equation of motion placed in this directory, named `motion.py`

Let us look at the parts of the script one-by-one.
```python
from scipy import integrate as itg
import numpy as np
import matplotlib.pyplot as plt
```

Here we import the Python libraries we will need. 

As was mentioned before `SciPy` offers an implementation of ODE solver.
We import the sub-package `integrate` and call it `itg`.

`NumPy` is probably the most used Python library since it offers multi-dimensional arrays and functions to manipulate them, along with many useful mathematical functions.

Finally, `matplotlib.pyplot` is a library for making plotting, here it will be used to display the trajectory.

```python
def derivs(y,t,E,B,q,m):
    d = np.zeros(np.shape(y))
    d[0:3] = y[3:6]
    d[3:6] = q/m*(E + np.cross(y[3:6],B))
    return d
```

This is the function which evaluates the first-order derivatives as functions of the *state variable*, here called `y`.
This variable contains the configuration variables *x,y,z* at indices 0,1,2 and velocities at 3,4,5.
The function also has contains the independent variable `t` which is not used in the body of the function.
I am using the `np.cross()` function to keep the code similar to the vector equations but you can write the equations for each component separately in your function if you want to.

```python
E = np.array([1.0,0.0,0.0])
B = np.array([0.0,0.0,1.0])
q = 1.0
m = 1.0

ti = 0.0
tf = 10.0
num_points = 1000
t = np.linspace(ti,tf,num_points)
y0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

res = itg.odeint(derivs,y0,t,args=(E,B,q,m))
```
This part configures the variables.
Please note, that the number of points is not affecting the quality of the result (or at least it should not), it only affects the output given.
The result in `res` should contain an array of 6-element arrays for each time in `t`.

```python
x = [ i[0] for i in res ]
y = [ i[1] for i in res ]
z = [ i[2] for i in res ]

plt.plot(y,x)
plt.show(block=True)
```
Now we only extract things we want from the result and create a plot.


# Van Allen radiation belt

![Van Allen](https://github.com/tungli/F5170-python/blob/master/3_Motion/van_allen.svg)

Earth's [Van Allen radiation belts](https://en.wikipedia.org/wiki/Van_Allen_radiation_belt) is a good example of charged particle moving in a field and suitable for numerical simulation.

## Exercises
>  **Exercise 4**
>  Let us assume that the magnetic moment of the Earth is accurately described by a single magnetic dipole moment and orient our frame of reference in such a way that this the magnetic moment is **_m_** = *(0,0,M)* in Cartesian coordinates.
>  * Express components of the magnetic field **_B_** in Cartesian coordinates.
>  * What is the value of *M* if the geomagnetic field at the equator is *3.12e-5* T?
>
>  **Exercise 5**
>  * Analyze the motion of the high-energy proton in the geomagnetic field. What are the three components of the motion?
>
>  **Exercise 6**
>  * Try changing the initial position of the proton
>  * What is the maximum initial distance from Earth's center for which the proton still has a stable (bound) trajectory?
>  * What is the minimum initial distance for which the proton does not hit the surface of the Earth?
>
>  **Advanced exercise**
>  * Replace the proton with an electron and try to find suitable initial conditions for a stable (bound) trajectory. Think before implementing it. Will the magnetic field required for an electron be higher or lower? How does the drift differ from that of the proton?

Since the same equations and solvers apply here, let us jump to the implementation right away.

## Implementation
The script is in this repository, named [van_allen.py](https://github.com/tungli/F5170-python/blob/master/3_Motion/van_allen.py).
You can see that the implementation is basically the same as with the particle drift study.
Some differences here are:
 - We assume electric field is zero
 - Earth's magnetic field can be approximated as if produced by a (huge) magnetic dipole.

```python
from scipy import integrate as itg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
```
Here some extra things are needed for 3D plots.

```python
def derivs(y,t,E,q,m,magM):
    d = np.zeros(np.shape(y))
    d[0:3] = y[3:6]
    d[3:6] = q/m*(E + np.cross(y[3:6],bfield(y[0:3],magM)))
    return d

def bfield(r,M):
    #This function calculates the magnetic field from 
    #position vector and magnetic dipole moment
    #Good luck!
```
We replace the magnetic field vector with a magnetic field vector generating function (which you are meant to write yourself).

```python
E = np.array([0.0,0.0,0.0])
magM = np.array([0.0,0.0,8.10e22])
q = 1.60217662e-19
m = 1.6726219e-27 
c = 3e8
re = 6.38e6
Ek_ev = 5e7

#Velocity
vr = c/np.sqrt(1.0+m*c**2/Ek_ev/np.abs(q))
vp = 0.0
vt = np.pi/4
v = np.array([vr*np.sin(vt)*np.cos(vp),vr*np.sin(vt)*np.sin(vp),vr*np.cos(vt)])
#Position
rr = 2.5*re
rp = 0.0
rt = np.pi/2
r = np.array([rr*np.sin(rt)*np.cos(rp),rr*np.sin(rt)*np.sin(rp),rr*np.cos(rt)])
#State vector
y0 = np.array([r[0],r[1],r[2],v[0],v[1],v[2]])

ti = 0.0
tf = 25.0
num_points = 10000
t = np.linspace(ti,tf,num_points)
```
Constants and initial values. Spherical coordinates for velocity and position vectors are used, which are then converted to Cartesian coordinates.
This is convenient because of the spherical nature of the Earth.


```python
res = itg.odeint(derivs,y0,t,args=(E,q,m,magM))

x = [ i[0]/re for i in res ]
y = [ i[1]/re for i in res ]
z = [ i[2]/re for i in res ]
```
Integration, followed by extraction of coordinates.

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_surface(xs, ys, zs, cmap=cm.plasma)
#Note: the colormap "plasma" is the only acceptable colormap when doing plasma physics!
#...although "jet" can be justified in some cases.
```
Plot the ball...

```python
ax.plot(x, y, z, lw=1.0, c="b")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
#ax.set_xlim((-3.0,3.0))
#ax.set_ylim((-3.0,3.0))
#ax.set_zlim((-3.0,3.0))

plt.show(block=True)
```
...and the trajectory.


