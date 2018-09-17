from scipy import integrate as itg
import numpy as np
import matplotlib.pyplot as plt

def derivs(y,t,E,q,m,magM):
    d = np.zeros(np.shape(y))
    d[0:3] = y[3:6]
    d[3:6] = q/m*(E + np.cross(y[3:6],bfield(y[0:3],magM)))
    return d

def bfield(r,M):
    #This function calculates the magnetic field from the magnetic moment
    l = np.linalg.norm(r)
    mr = np.dot(m,r)
    return 1e-7*(3*mr/l**5*r - m/l**3)

E = np.array([1.0,0.0,0.0])
magM = np.array([0.0,0.0,1.0])
q = 1.0
m = 1.0

ti = 0.0
tf = 10.0
num_points = 1000
t = np.linspace(ti,tf,num_points)

y0 = np.array([1.0,1.0,0.0,0.0,0.0,0.0])
res = itg.odeint(derivs,y0,t,args=(E,q,m,magM))

x = [ i[0] for i in res ]
y = [ i[1] for i in res ]
z = [ i[2] for i in res ]

plt.plot(y,x)
plt.show(block=True)
