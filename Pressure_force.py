import Viz_ToCCo as Viz
import sys
from sympy import *
from sympy.vector import CoordSys3D, curl, gradient, divergence, Del, Divergence, Gradient, laplacian, Curl, matrix_to_vector
import numpy as np
import pickle
def taylor(exp, nv, nt, dic):

    if nv == 0 and nt == 0:
        expt = exp.xreplace({ev: 0, et: 0})
        return(expt)
    if nt == 0:
        df = diff((exp.xreplace(dic).doit()), ev, nv).xreplace({ev: 0, et: 0})
    elif nv == 0:
        df = diff((exp.xreplace(dic).doit()), et, nt).xreplace({ev: 0, et: 0})
    else:
        df = diff((exp.xreplace(dic).doit()), ev, nv,
                  et, nt).xreplace({ev: 0, et: 0})

    expt = (1 / (factorial(nt) * factorial(nv))) * df
    return expt

def taylor_serie(exp, nvmax, ntmax, dic):
    nv = np.arange(nvmax + 1)
    nt = np.arange(ntmax + 1)
    expt = 0
    for v in nv:
        for t in nt:
            tay = taylor(exp, v, t, dic) * et**t * ev**v
            expt += tay
            print(str(v+t) + '/' + str(nvmax+ntmax),end = '\r')
    return(expt)

def meanterm(Phi):
    APhi = np.array(list(expand(Phi).args))
    symba = (np.array([t not in a.free_symbols for a in APhi]) *
             np.array([x not in a.free_symbols for a in APhi]) *
             np.array([y not in a.free_symbols for a in APhi])
             )
    if symba.size > 0:
        Phi2 = sum(Matrix(APhi[symba]))
    else:
        Phi2 = 0
    return(Phi2)

def simp(expr):
    expe = powsimp(expand(expr))
    return(expe)

def conjv(vect):
    ve = matrix_to_vector(((vect).to_matrix(C).H).xreplace({conjugate(x): x, conjugate(
        y): y, conjugate(z): z, conjugate(t): t, conjugate(et): et, conjugate(ev): ev}), C)
    return(ve)

def conj(scal):
    ve = conjugate(scal).xreplace({conjugate(x): x, conjugate(y): y, conjugate(
        z): z, conjugate(t): t, conjugate(et): et, conjugate(ev): ev})
    return(ve)

def realve(vect):
    ve = (vect + conjv(vect)) / 2
    return(ve)

def surfcond(val, dic):
    va = val.xreplace(dic).doit().xreplace(
        {z.xreplace(dic): -(f - z)}).xreplace(dic)
    return(va)

filename = sys.argv[1]
C = CoordSys3D('C')
x = C.x
y = C.y
z = C.z
x.is_real
y.is_real
z.is_real
x._assumptions['real'] = True
y._assumptions['real'] = True
z._assumptions['real'] = True
t = Symbol('t')
et,ev = symbols('et,ev',real = True)
qRm,qRmm,Rl,qRo = symbols('qRm,qRmm,Rl,qRo')

with open(filename, 'rb') as f:
    data = pickle.load(f)
meta = data['meta']
solfull = data['Expr']

print('################################################################')
print("Computation of Pressure torque for ", filename)
print('################################################################')

condB = meta['condB']
orders = np.array(meta['order_list'])
print(orders)
B = solfull[4]*C.i+solfull[5]*C.j+solfull[6]*C.k
p = solfull[3]
spsi  = solfull[8]


f = z - data['topo']  # +et**2 *f2
print(f)

delf = gradient(f)
n = delf.normalize()

Bt = simp(realve(B))
P = simp(((conj(p) + p) / 2))
nn = simp(realve(n))
Fp = surfcond(simp((P * nn)), {})
Fpm = surfcond(simp((((Bt.dot(Bt)) / 2) * nn)), {})

Fptay = taylor_serie((Fp & C.i) - (Fpm & C.i),np.max(orders[:,0]) * 2, np.max(orders[:,1])*2,{})
ValPress = meanterm(expand((Fptay)))


rhoscale = np.float64(1e4) #scale considering rho = 1e4
Xscale = np.float64(meta['Rl']*2890000)
Vscale = np.float64(7.29e-5*Xscale*meta['Ro'])

print('\n')
print("Pressure torque : ", ValPress)
ValPress_D = ValPress.xreplace({et:Symbol('h')/Xscale,ev:1})*rhoscale*Vscale**2
print("Dimensional Pressure torque : ", ValPress_D," N/m^2")

try:
    zeta_value = np.float64(sys.argv[2])
    print('\n ####################################### \n')
    print("Pressure torque for "+str(np.round(zeta_value,5))+" zeta :", re(ValPress.xreplace({et:zeta_value})))
    print("Dimensional Pressure torque for a topo of "+str(np.round(zeta_value*Xscale,5))+" m :", re(ValPress_D.xreplace({Symbol('h'):zeta_value*Xscale}))," N/m^2")
except:
    pass
#Dimensional diss
