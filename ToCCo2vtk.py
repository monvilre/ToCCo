import numpy as np
from pyevtk.hl import gridToVTK
from mpmath import *
from sympy import * 
from sympy import E, Eq, Function, pde_separate, Derivative as D, Q
from sympy.vector import CoordSys3D
from sympy.vector import CoordSys3D,matrix_to_vector,curl,gradient,divergence,Del,Divergence,Gradient, laplacian,Curl
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr

C = CoordSys3D('C')
init_printing()
U0,omega,qRe,chi,qRo,BOx,qAl,BOz,BOy,qRm,qFr,Ri,zeta,Dist,e,alpha,t = symbols("U0,omega,qRe,chi,qRo,BOx,qAl,BOz,BOy,qRm,qFr,Ri,zeta,Dist,e,alpha,t")
x = C.x
y = C.y
z = C.z
# Fields to convert
fil = open("sol_topo","r")
fi = fil.read()
fil.close()

ind = fi.find("Matrix")
Mat = fi[ind:]
exec("solfull =" + Mat)
dic = fi[:ind]
exec("dic =" + dic)
zet = np.float(dic[zeta])
solfull = solfull.xreplace({t:0})
ff = lambdify((x,y,z),solfull,dummify = True)

#Change space limit here
x = np.linspace(-2*np.pi,2*np.pi,50, dtype='float64')
y = x
z = np.linspace(0,-np.pi,20, dtype='float64')

z = np.copy(z, order = 'F')
x = np.copy(x, order = 'F')
y = np.copy(y, order = 'F')
data = np.zeros((len(x),len(y),len(z),7), dtype='float64')
for d in range(0,7):
	print(d)
	for i,xx in enumerate(x):
		for j,yy in enumerate(y):
			for k,zz in enumerate(z):
				data[i,j,k,d] = np.float64(np.real(ff(xx,yy,zz)[d][0]))
print(np.shape(data[:,:,:,0]))
data = np.copy(data, order = 'F')
gridToVTK("./Vtk_topo",x,y,z, pointData = {"U":data[:,:,:,0],"V":data[:,:,:,1],"W":data[:,:,:,2],"P":data[:,:,:,3],"Bx":data[:,:,:,4],"By":data[:,:,:,5],"Bz":data[:,:,:,6]})
print('Done')
