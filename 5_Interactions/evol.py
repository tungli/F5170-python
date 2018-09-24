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

