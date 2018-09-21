import numpy as np
import matplotlib.pyplot as plt

colors = ['r','g','b','m','k','y','c']

x = np.linspace(0,1,1000)
for i in range(0,4):
    plt.plot(x,x**i,colors[i],label="x^" + str(i))

plt.legend()
plt.show(block=True)

