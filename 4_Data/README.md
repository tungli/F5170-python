# Data processing

In this chapter, you will learn to use *for*-loops and *if* statement blocks.

You will understand how to do those just by going over the examples, so I will not go into detail here.
When writting statements in Python just **do not forget indenting and `:` signs**.

Let us take a look at the scrips you will need for the exercises.

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
In the beginning we import our favourite packages.
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
It notifies us when the matrix is not square or when its determinant is zero and in these cases, it does not try to call the matrix inversion funcion.
Couple of things to notice:
 * `np.shape()` returns a `tuple` (an object similar to `list`) with the array's size.
 * `if`,`elif` are followed by a condition and a `:` symbol. A condition has a value of `True` or `False`.
 * Indenting is used, just as in for-loops and functions
 * `return` keyword exits the function with the desired value.

 

