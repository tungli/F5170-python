# Data processing
In this chapter, you will learn to use *for*-loops and *if* statement blocks and the basics of file manipulation.

You will understand how to do those just by going over the examples, so I will not go into detail here.
When writing statements in Python just **do not forget indenting and `:` signs**.


The data files you will be working here (`csk[1-3].dat`) contain data for the cross sections of collision (in m<sup>2</sup>) of electrons with argon atom species for different temperatures (in Kelvin).

In particular, `csk1.dat` contains the data for elastic collision:

![coll](http://mathurl.com/ycnhzk89.png)

`csk2.dat` for argon excitation:

![exci](http://mathurl.com/ybd2s6ql.png)

`csk3.dat` for argon ionization:

![ioni](http://mathurl.com/ydyupuzm.png)

The scripts referenced in the exercises are described in subsequent sections.

## Exercises
>  **Exercise 1**
>  * Modify [this script](https://github.com/tungli/F5170-python/blob/master/4_Data/simple_plot.py) so that it plots:
>  
>  ![sinc](http://mathurl.com/y983ysyp.png)
>  
>  **Exercise 2**
>  [This script](https://github.com/tungli/F5170-python/blob/master/4_Data/data_plot.py) takes the data in `csk[1-3].dat` and plots them.
>  * Run the script with the data in the same directory
>  * You can see that the magnitudes are too different and the plot is not very practical. Your task is to modify it so that:
>    1. The electron energy is expressed in electronvolts (eV).
>    2. The y-axis scale is logarithmic
>  
>  The logarithmic plot should look similar (hopefully with better axes) to this:
>  ![Data](https://github.com/tungli/F5170-python/blob/master/4_Data/data_plot.svg)
>  
>  * Look at your plot -- what are the excitation and ionization thresholds?
>  
>  **Exercise 3**
>  * Modify the script so that it converts the *x*-data from Kelvin to electronvolt and saves each cross-section to tab-delimited file named `csevN.dat` with `N` being the corresponding file number.
>  
>  **Exercise 4**
>  * Find the function in Python which calculates the inverse of a (NumPy) matrix. 
>  * Define a [singular matrix](http://mathworld.wolfram.com/SingularMatrix.html) and try calculating its inverse. What happens?
>  * Run [this script](https://github.com/tungli/F5170-python/blob/master/4_Data/inverse_matrix.py) and verify that it works correctly by testing it on various matrices.
>  * Add another `elif` statement so that it displays a warning when the [matrix rank](http://mathworld.wolfram.com/MatrixRank.html) is greater than 10.
>  
>  **Advanced exercise**
>  In this directory you will also find data files `adv_csk[1-3].dat`. Two of these files include electron temperature in electronvolts (eV) while one of them includes electron energy in Kelvin.
>  * Modify the [script](https://github.com/tungli/F5170-python/blob/master/4_Data/data_plot.py) using *if-else* statements so that it decides which files should be converted and which not.
>  * Such a script could be very useful when processing thousands of similar files. However, what are the limitations of your program?


Let us take a closer look at the scrips you will need for the exercises.

## Simple plot
```python
import numpy as np
import matplotlib.pyplot as plt
import antigravity

colors = ['r','g','b','m','k','y','c']
x = np.linspace(0,1,1000)

for i in range(0,4):
    plt.plot(x,x**i,colors[i],label="x^" + str(i))

plt.legend()
plt.show(block=True)
```
In the beginning we import our favorite packages.
Then, we define a `list` of `string`s.
These are arguments used in `plot()` to define the color of the line.
Next, we define a NumPy array for the plot.

Then we create a for-loop which iterates over the `list` (in this case `[0,1,2,3]`) created by the `range()` function.
Inside the for-loop we make a simple plot with a legend.

**Note:** Here is one other way we could have designed the for loop -- using the `enumerate()` function.
```python
for i,c in enumerate(colors):
    plt.plot(x,x**i,c,label="x^" + str(i))
```
After the for-loop we tell Python it can show us the figure.

## Loading and plotting a matrix
```python
import numpy as np
import matplotlib.pyplot as plt

c = ['r','g','b','m','k','y','c']
for i in range(0,3):
    filename = 'csk' + str(i+1) + '.dat'
    f = open(filename)
    d = f.read()
    f.close()
    
    data = d.replace('\t','\n').split('\n')[:-1:]
    x = np.array([float(j) for j in data[0::2]])
    y = np.array([float(j) for j in data[1::2]])

    plt.plot(x,y,c[i],label=filename)
    plt.xlim(0,1e6)

plt.legend()
plt.show(block=True)
```
In this script, we first import packages, then define a color vector, then we load the files *csk1.dat, csk2.dat* and *csk3.dat*, extract the data and plot it using a for-loop.

There is no universal way to import data from files into Python, as far as I know - files are usually too different.
Here we use `open()`-`read()`-`close()` sequence to get to the file into a string variable.
Then we manipulate the string, particularly, we use the `str.replace()` and `str.split()` methods to separate the values and then apply indexing to group the columns together.
We use [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) instead of initializing arrays and adding values in a *for*-loop to keep the code short.


## Saving data

The easiest way to save maxtrix-like data would be to make use of NumPy's [savetxt function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html).

The hard way can be useful in some situations, so here is an example.
Consider the arrays `x`, `y` from the script above.
These arrays should have the same length.
We could iterate through indices of these arrays and place each pair of values into a string which in turn will become one line of the file.
Here is an example script:

```python
filename = 'new_file.data'
line = []

for i in range(len(x)):
    line.append('{}, {}'.format(x[i], y[i]))

file_content = '\n'.join(line)

f = open(filename, 'w')
f.write(file_content)
f.close()
```

Two things to comment here:
* We are making use of several string and list methods:
   - `append` adds an element to the end of an already-existing list.
   - `format` method which fills in a *template* string with its values. Notice that the conversion from numeric type to string is implicit (no need to call `str(x[i]`).
   - `join` joins list elements to a string with '\n' in between.
* In function `open`, the second argument means "open for writing". **This will overwrite the contents of the file, so be carefull with your file name.**


## Inverse matrix
```python
import numpy as np

def myinv(mat):
    s = np.shape(mat)
    if s[0] != s[1]:
        print("Not a square matrix!")
        return None
    elif np.linalg.det(mat) == 0.0:
        print("Determinant is zero!")
        return None
    else:
        return np.linalg.inv(mat)
```
We created a more sophisticated version of matrix inversion.
It notifies us when the matrix is not square or when its determinant is zero and in these cases, it does not try to call the matrix inversion function.
Couple of things to notice:
 * `np.shape()` returns a `tuple` (an object similar to `list`) with the array's size.
 * `if`,`elif` are followed by a condition and a `:` symbol. A condition has a value of `True` or `False`.
 * Indenting is used, just as in for-loops and functions
 * `return` keyword exits the function with the desired value.
