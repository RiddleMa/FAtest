#encoding=utf8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=Axes3D(fig)
X=np.arange(-5.12,5.12,0.1)
Y=np.arange(-5.12,5.12,0.1)
X,Y=np.meshgrid(X,Y)
Z=20+X*X+Y*Y-10*np.cos(2*np.pi*X)-10*np.cos(2*np.pi*Y)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.cm.hot)
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap=plt.cm.hot)#这句是X-Y平面加投影的
ax.set_zlim(0,100)
plt.show()