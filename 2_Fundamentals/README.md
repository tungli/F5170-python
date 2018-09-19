# Extra problems

## Vector product
Write a function that accepts 5 arguments: *q,m,v,E,B*, and calculates the Lorentz force.
Browse the internet to find the function which calculates the [vector product](https://en.wikipedia.org/wiki/Cross_product) of two 3-element vectors and implement it in your function.
Test your function. 

## Debye length
One of the important parameters of a plasma is the [Debye length](https://en.wikipedia.org/wiki/Debye_length).
In plasma it can be calculated from electron number density and electron temperature:

![Debye](http://mathurl.com/y876kcbb.png)

First, implement a function which calculates the Debye length.
Then, download the `Discharges.txt` file. 
Ideally, you want to have the file in the same directory where you are working.

If working in MATLAB:
 * Use the *Home -> Import Data* graphical interface. 
 * After selecting what you want to import and confirming *Import Data* you will create a variable that you can work with further
 * Inspect this variable by double-clicking at it in the *Workspace* window.
 * Write the name of the variable followed by a dot and press Tab. Can you extract a single column from the table?

If working in Python consider the following few lines of code:
```python
f = open('Discharges.txt')
a = f.read().replace('\n',' ').replace('  ',' ').split(' ')
f.close()
b = [a[0::3][1:-1],a[1::3][1::]]
```
 * What is stored in variable `b`? 
 * Can you understand what each line of the code does? Try running each line one-by-one and printing the contents of it. Try changing the indices in the last line. You can also check out [this page](https://www.pythoncentral.io/cutting-and-slicing-strings-in-python/).
 * Can you create a list which contains the plasma names and their temperature?  

Now use your function to calculate the Debye lengths of all plasmas with only one function call.

## Larmor radius
What will be the trajectory of a charged partice with an initial velocity in homogeneous magnetic field?
Try to guess what the following parameter, called *Larmor radius*, represents in the trajectory:

![Larmor](http://mathurl.com/ybs37jkj.png)

Download the `result.txt` file which contains the data from such trajectory.
In this file there are 3 columns -- 1. column is time, 2.column is the *x*-position and 3. column the *y*-position.

If working in MATLAB:
 * Search the internet for the documentation of the `dlmread()` function

If working in Python:
 * Search the internet for the documentation of the `loadtxt()` function from the NumPy package

Try it out on the file.

Now, plot *x* as a function of time and *y* as a function of *x*.

Suggest how to get an estimate of the Larmor radius from the data.
Implement the method.  

Assume the charged particle in the trajectory was an electron with an intial velocity of 1 m/s.
Calculate the magnetic field.

