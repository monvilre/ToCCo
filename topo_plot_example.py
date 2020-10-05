import numpy as np
from mpmath import *
from sympy import * 
from sympy import E, Eq, Function, pde_separate, Derivative as D, Q
from sympy.vector import CoordSys3D
from sympy.vector import CoordSys3D,matrix_to_vector,curl,gradient,divergence,Del,Divergence,Gradient, laplacian,Curl
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr


C = CoordSys3D('C')
init_printing()
U0,omega,qRe,chi,qRo,BOx,qAl,BOz,qRm,qFr,Ri,zeta,Dist,e = symbols("U0,omega,qRe,chi,qRo,BOx,qAl,BOz,qRm,qFr,Ri,zeta,Dist,e")
x = C.x
y = C.y
z = C.z


fil = open("./sol_topo","r")
fi = fil.read()
fil.close()
ind = fi.find("Matrix")
Mat = fi[ind:]
exec("solfull =" + Mat)
dic = fi[:ind]
exec("dic =" + dic)
zet = np.float(dic[zeta])
print(dic)


ff = lambdify([x,z],solfull.xreplace({y:0}),'numpy')
xn = np.linspace(-2*np.pi,2*np.pi,200)
yn = np.linspace(0,-1,200)
xqui = xn[::6]
yqui = yn[::6]
Xqui,Yqui = np.meshgrid(xqui,yqui)
X,Y = np.meshgrid(xn,yn)

disp = 0
print(ff(2,0))
var = ["ux","uy","uz","p","bx","by","bz"]
plt.figure(figsize=(15,8))
plt.plot(xn,np.real(zet*np.exp(1j*xn)),'k')
#plt.plot(xn,np.real(zet*np.exp(1j*xn))-1,'k')
plt.contourf(X,Y,np.real(ff(X,Y))[disp][0],200,cmap = "viridis",zorder=1)
plt.colorbar()
#plt.streamplot(X,Y,np.real(ff(X,Y)[0][0]),np.real(ff(X,Y)[2][0]),color = "k",linewidth = 0.5, density = 4,arrowsize = 0.5,maxlength = 6)
plt.quiver(Xqui,Yqui,np.real(ff(Xqui,Yqui)[0][0]),np.real(ff(Xqui,Yqui)[2][0]),zorder=2)
plt.show()
