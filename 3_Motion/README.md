# Particle drift
This sections covers exercises 3.1, 3.2, 3.3 and 3.1.a.

## Matematical formulation

The exercises want us to solve the equation of motion for a charged particle in a electromagnetic field.
The equation of motion is a set of 3 (x,y,z) ordinary differential equations (ODE) of second order.
To solve a system of ODEs, we usually transform the set of second order equations to first order.
We will create more variables by doing this but that's fine.
Then we express the first order derivatives as functions of variables and other first order derivatives.
We can then create a function in Python which for a given set of values of variables evaluates the derivatives.
This is the function that an ODE solver (usually) needs.

In our problem we treat positions and velocities as variables and express their derivatives as functions of these variables:

![Equation of Motion](http://mathurl.com/y79nmta7.png)

where *F* is the Lorentz force.

**Note:** There is another way! Here a general approach to ODE is shown but to integrate second order ODEs it is usually not necessary to transform to first order set. 
Another way is, for example, to use the *leapfrog* algorithm. 
If you are not comfortable with using a blackbox solver, this is the way for you. Try googling "Boris leapfrog" and implementing it in your code.

## Implementation
In this implementation we will be using the `odeint` solver from the `scipy.integrate` subpackage which is a Python wrap around the *ODEPACK* library written in Fortran.
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
We import the subpackage `integrate` and call it `itg`.

`NumPy` is probably the most used Python library since it offers multi-dimensional arrays and functions to manipulate them, along with many useful mathematical functions.

Finally, `matplotlib.pyplot` is a library for making ploting, here it will be used to display the trajectory.

```python
def derivs(y,t,E,B,q,m):
    d = np.zeros(np.shape(y))
    d[0:3] = y[3:6]
    d[3:6] = q/m*(E + np.cross(y[3:6],B))
    return d
```

This is the function which evaluates the first-order derivatives as functions of the *state variable*, here called `y`.
This variable contains the configuration variables $x,y,z$ at indices 0,1,2 and velocities at 3,4,5.
The function also has contains the independent variable `t` which is not used in the body of the funtion.
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










