# Learning Python

Python is considered a simple language (compared to the rest) and well-suited for numerical computing.
This short tutorial should help you get started with Python.

**Warning:**
I will not be concerned about differences between Python 2 and Python 3 so use Python 3 (possibly a newer version).

# Using Python

Unlike with Matlab, there are multiple ways to write code in Python.
In my humble opinion the most headache-avoiding way is to use a simple text editor with syntax highlighting (for Windows there is [Notepad++](https://notepad-plus-plus.org/), for Linux Gedit, Vim, Emacs) and [IPython](https://ipython.org/).
With these tools you can write Python programs in editor and test individual lines of code in IPython or even run the whole script (using `%run your_python_script.py`).

A second approach is to work in a Python IDE ([Spyder](https://www.spyder-ide.org/) or [PyCharm](https://www.jetbrains.com/pycharm/).
There are benefits to IDEs -- they come with a huge amount of features and are specialized, usually, for one language, but you really the only thing you need to write code is a text editor (at least for these exercises) and IDEs can distract you from appreciating how simple programming in Python really is.

Third way, and among scientists probably the most used one, is to use [Jupyter](https://jupyter.org/) notebook.
Jupyter notebook is really amazing and very nice for beginners but I think it encourages a copy-paste-driven developement style of programming and gives and makes you more lazy towards abstracting your code and exporing advanced Python features (classes, modules).
That being said, for the scope of this course Jupyter notebook may be the perfect tool.

Pick one or try several to see which one suits you the best.

# Python variables, data types, operators, print, lists...

For the long version read [this](https://pythonbasics.org/) (or any other tutorial).

In Python you do not declare types of variables explicitely.
The following is a valid Python code:

```python
a = 2
a = "two"
a = 2.0
a = True
```

Here the variable `a` is overwritten with different datatypes (integer, string, floating-point number, boolean).
You can sum an integer and a float without a problem:

```python
2 + 3.0 # == 5.0
```
The parts of code after `#` are comments (they are ignored by python but should not be ignored by the programmer).

To make Python display use the `print` function:

```python
print(2 + 3.0)
```

Data types in Python are of two types: immutable and mutable.

Immutable means that their values cannot be changed.
Strings and numbers are immutables so let's introduce a mutable type to see the difference.
A `list` is a mutable data type which stores any data:

```python
a = [] # an empty list
b = [1, 2, 3] # a list of numbers
c = [1, 'two', 3.0] # a list with heterogeneous elements
d = [c, 1, 2] # a list inside list plus other elements
```

You can get an element of list by indexing:

```python
c = [1, 'two', 3.0]
c[0] # == 1
c[1] # == 'two'
```

and change an element similarly:

```python
c = [1, 'two', 3.0]
c[0] = 0
c[1] = 1
print(c)
```

This is different with numbers or strings.
For example, you cannot change a string:

```python
a = "Easilz"
a[5] = "y" # will not work
```

and when you do this:

```python
a = 3
a = 4
```
you are not changing the value `3` to `4`.
In this case you are only making the variable `a` to point to value `4` (and the value `3` is no longer accessible).

# Functions

The best thing about programming is the ability to abstract things out.
Functions do precisely that.

Take this for example: you are given 3 3D-vectors and you want to calculate their (physical) lengths.
You can do simply:

```python
a = [1,2,3]
b = [4,1,5]
c = [3.0, 3.0, 3]

length_of_a = a[0]*a[0] + a[1]*a[1] + a[2]*a[2]
length_of_b = b[0]*b[0] + b[1]*b[1] + b[2]*b[2]
length_of_c = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
```

This gets the job done but it is error-prone and lengthy and if you had more vectors, really tedious.
**Never write the same code twice!**
What you can do in this case is to define a function:

```python
def length_of(x):
    length = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    return length
```

Here `length_of` is the name of the function, `x` is an argument of the function (a function can take multiple arguments) and `length` is *returned* from the function.
You can then call the function on an object like this:

```python
length_of(a)
```

Write functions often and do not write functions that are too long.
It will help you to become a good programmer and to write simple, elegant and testable code.


# More on lists

You will make use of lists often.
Lists have built-in functionality in form of methods.

A method is like a function that belong to a particular object.
It is called with a dot `.`, its name and parentheses `()` (possibly with arguments):

```python
a = [1, 2, 3]
a.reverse()
print(a)

a.extend([4,5,6])
print(a)
```

[Here](https://www.tutorialspoint.com/python/python_lists.htm) is the complete list of methods.

**Note:**
Strings and other objects also have methods that you might find useful.

# Python libraries

The greatest thing about Python is that it is a very popular language.
The problems you will face will almost surely have been solved by other people and a simple google search will give you the solution.
For common problems, there are Python libraries ready for use.

For example, the `NumPy` library is a collection of numerical methods.
We will use this a lot.

Other libraries useful for numerical computing include `SciPy`, `Matplotlib`, `Pandas`.

To use a library, you will first have to download it and install it where Python can find it.
Then, inside of a program you have to reference that you are using the library/module.
The keyword for this is `import`:

```python
import numpy
```

Since `numpy` is too long we can name it `np`:

```python
import numpy as np
```

Then we are able to use any function or other object inside numpy:

```python
s = np.sum([1,2,3]) # == 6
```

## NumPy arrays

For numerical computing `list`s are not too useful.
For one thing they are slow.
For another:

```python
a = [1,2,3]
print(3*a)
```

For numerical stuff, NumPy's "lists" are better:

```python
a = np.array([1,2,3])
print(3*a)

b = np.array([2,3,4])
print(2*a + b)
```

## Matplotlib

To generate pretty pictures we will use `matplotlib`:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5])
plt.plot(x, x**2, label="first")
plt.xlabel('Numbers')
plt.ylabel('Superior numbers')

plt.plot(x, np.sin(x), '-.', label="second")
plt.legend()
plt.show()
```

# Conclusion

I am aware that this is not in any sense a good introduction to Python but hopefully you have some intuition of how things work and you can figure out the rest with the help of google.

If not, do not despair, the exercises in other chapters provide ample guidance.
If you get stuck, it is probably my fault, so complain -- write me an email or submit an issue on github.

# Exercises


Complete the problems 2.1 and 2.2 in the [material for MATLAB](https://github.com/tungli/F5170-matlab/blob/master/MatlabSkriptaEN.pdf)
You can then try solving the following problems.

## Vector product
Write a function that accepts 5 arguments: *q,m,v,E,B*, and calculates the acceleration from the Lorentz force.
Browse the internet to find the function which calculates the [vector product](https://en.wikipedia.org/wiki/Cross_product) of two 3-element vectors and implement it in your function.
Test your function. 

## Debye length
One of the important parameters of a plasma is the [Debye length](https://en.wikipedia.org/wiki/Debye_length).
In plasma it can be calculated from electron number density and electron temperature:

![Debye](http://mathurl.com/y876kcbb.png)

Implement a function which calculates the Debye length.
Then, download the `Discharges.txt` file. 
Ideally, you want to have the file in the same directory where you are working.

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

Now use your function to calculate the Debye lengths of all plasmas.

## Larmor radius
What will be the trajectory of a charged partice with an initial velocity in homogeneous magnetic field?
Try to guess what the following parameter, called *Larmor radius*, represents in the trajectory:

![Larmor](http://mathurl.com/ybs37jkj.png)

Download the `result.txt` file which contains the data from such trajectory.
In this file there are 3 columns -- 1. column is time, 2.column is the *x*-position and 3. column the *y*-position.

Search the internet for the documentation of the `loadtxt()` function from the NumPy package
Try it out on the file.

Now, plot *x* as a function of time and *y* as a function of *x*.

**Advanced:**

Suggest how to get an estimate of the Larmor radius from the data.
Implement the method.  

Assume the charged particle in the trajectory was an electron with an intial velocity of 1 m/s.
Calculate the magnetic field.

