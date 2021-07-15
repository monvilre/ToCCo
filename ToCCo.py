#! /usr/bin/env python3
# coding: utf-8

from sympy import *
from sympy import Function, Derivative as D
from sympy.vector import CoordSys3D, curl, gradient, divergence, Del, Divergence, Gradient, laplacian, Curl, matrix_to_vector
# from sympy.solvers.solveset import linsolve
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import mpmath as mp
import params
from sympy import symbols, Function, Eq
import itertools
from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type, _classify_linear_system, _linear_ode_solver, matrix_exp_jordan_form
import scipy.integrate as integ
import pickle

# from sympy.solvers.ode.subscheck import checkodesol

mp.mp.dps = params.prec
C = CoordSys3D('C')

#######################
###    FUNCTIONS    ###
#######################


def mpmathM(A):
    """Convert Sympy Matrix in mpmath Matrix format.

    Attrs:
    - A (Sympy Matrix): Matrix to convert.

    Returns:
    - An mpmath Matrix.
    """
    A = A.evalf(mp.mp.dps)
    B = mp.matrix(A.shape[0], A.shape[1])
    for k in range(A.shape[0]):
        for l in range(A.shape[1]):
            B[k, l] = mp.mpc(str(re(A[k, l])), str(im(A[k, l])))
    # return(mp.chop(B,tol = 10**(-mp.mp.dps/3)))
    return(B)  # ,tol = 10**(-mp.mp.dps/1.2)))
    # return(B)


def factorial(n):
    """ Return the factorial of n """
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)

#
# def printn(expr, n=3):
#     try:
#         expU = (expr).find(exp)
#         logexpU = [exp(N(log(h).expand(), n)) for h in expU]
#         print(N((expr).xreplace(dict(zip(expU, logexpU))), n))
#     except:
#         print(expr)


def findexp(Ma):
    def coe(x): return list(x.find(exp))
    expo = (Ma.applyfunc(coe))
    expo = set([val for sublist in expo for val in sublist])
    return(expo)


def makedic(eig, order):
    dic = {
        Symbol('u' + str(order) + 'x'): eig[0],
        Symbol('u' + str(order) + 'y'): eig[1],
        Symbol('u' + str(order) + 'z'): eig[2],
        Symbol('p' + str(order)): eig[3],
        Symbol('b' + str(order) + 'x'): eig[4],
        Symbol('b' + str(order) + 'y'): eig[5],
        Symbol('b' + str(order) + 'z'): eig[6],
        Symbol('rho' + str(order)): eig[7]
    }
    return(dic)


def strain(U):
    xy = 1 / 2 * (diff(U & C.i, y) + diff(U & C.j, x))
    xz = 1 / 2 * (diff(U & C.i, z) + diff(U & C.k, x))
    yz = 1 / 2 * (diff(U & C.j, z) + diff(U & C.k, y))
    strainT = Matrix([[diff(U & C.i, x), xy, xz],
                      [xy, diff(U & C.j, y), yz],
                      [xz, yz, diff(U & C.k, z)]])
    return(strainT)


def simp(expr):
    expe = powsimp(expand(expr))
    return(expe)


def comat(vec, ex):
    def coe(x): return x.coeff(ex) * ex
    M = (vec.to_matrix(C))
    M1 = M.applyfunc(coe)
    vec1 = matrix_to_vector(M1, C)
    return(vec1)


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


def null_space(A):
    u, s, vh = mp.svd_c(A)
    # /4 is the most important tolerance parameters for the number of kz
    for inc in reversed(range(8)):
        if mp.almosteq(s[inc], 0, (10**(-(mp.mp.dps / 3)))) == True:
            val = sympify(vh[inc, :].transpose_conj())
            Btest = (val[4] * C.i + val[5] * C.j + val[6] * C.k)
            divss = N(simplify((divergence(
                Btest * ansatz.xreplace({kz: solrnow}))) / ansatz.xreplace({kz: solrnow})))
            if mp.almosteq(divss, 0, (10**(-(mp.mp.dps / 3)))) == True:
                # if mp.almosteq(divss, 0, 1e-10) == True:
                # verif = (A*(vh[inc,:].transpose_conj()))
                # print('verif',mp.chop(verif))
                # print('ansatz1',ansatz)
                # return(mp.chop(vh[inc, :].transpose_conj(),tol =10**(-(mp.mp.dps / 3))) )
                return(vh[inc, :].transpose_conj())

                break
    return([mp.nan])

# This is an improved version of the taylor term computation of sympy


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
            print('order', v, t)
    return(expt)


def makeEquations(U, B, p, rho, order_v, order_t, dic):
    Eq_rho = diff(rho, t) + (U.dot(gradient(rho)))
    if buf == 1:
        Eq_rho = diff(rho, t) + (U.dot(diff(rho0, z) * C.k))
    buoy = rho * qFr**2 * g / g0
    Cor = ((2 * qRo * (mp.sin(LAT) + Rl * mp.cos(LAT) * y)) * C.k).cross(U)
    # Cor = (2*qRo*(mp.sin(LAT))* C.k).cross(U)
    BgradB = (AgradB.xreplace({Ax: B & C.i, Ay: B & C.j, Az: B &
                              C.k, Bx: B & C.i, By: B & C.j, Bz: B & C.k})).doit()
    UgradU = (AgradB.xreplace({Ax: U & C.i, Ay: U & C.j, Az: U &
                               C.k, Bx: U & C.i, By: U & C.j, Bz: U & C.k})).doit()
    if buf == 1:
        BgradB = 1e-10 * BgradB
        UgradU = 0 * C.i
    if atmo == 1:
        BgradB = 1e-10 * BgradB
    if params.condU == 'Inviscid':
        Eq_NS = diff(U, t) + Cor + UgradU - (-gradient(p) + buoy + BgradB)
    else:
        Eq_NS = diff(U, t) + Cor + UgradU - (-gradient(p) + qRe * laplacian(U) + buoy + BgradB)

    Eq_vort = diff((Eq_NS & C.j), x) - diff((Eq_NS & C.i), y)

    Eq_m = divergence(U)

    if buf == 1:
        Eq_b = diff(B, t) - (qRm * diff(B, z, z)) - qAl * diff(U, z)
    if atmo == 1:
        Eq_b = diff(B, t) - (qRm * laplacian(B) + 1e-8 * (curl(U.cross(B))))
    else:
        Eq_b = diff(B, t) - (qRm * laplacian(B) + (curl(U.cross(B))))
    eq = zeros(9, 1)

    for ii, j in enumerate([Eq_NS & C.i, Eq_NS & C.j, Eq_vort, Eq_NS & C.k, Eq_b & C.i, Eq_b & C.j, Eq_b & C.k, Eq_m, Eq_rho]):
        eq[ii] = taylor(j, order_v, order_t, dic)
    eq = Matrix([powsimp(expand(eqx).xreplace(dic)) for eqx in eq])

    print("Eq OK")
    return(eq)


def eigen(M, dic):
    M1 = M.xreplace(dic)
    # M1 = N(M1,mp.mp.dps * 10)
    # M2 = np.array(M1)
    # Mde = Matrix(M2[np.ix_(np.any(M2 != 0, axis=1), np.any(M2 != 0, axis=1))])
    Mde = M1
    # Mde = Matrix(M1)
    det = (Mde).det(method='berkowitz')
    detp = Poly(det, kz).xreplace(dic)
    co = detp.all_coeffs()
    co = [N(k, mp.mp.dps * 10) for k in co]

    maxsteps = 3000
    extraprec = 5000
    co = [mp.mpmathify(coco) for coco in co]
    sol, err = mp.polyroots(
        co, maxsteps=maxsteps, extraprec=extraprec, error=True)
    sol = np.array(sol)

    print("Error on polyroots =", mp.chop(err))
    solr = sol
    # solr = solr[[mp.fabs(m) < 10**(mp.mp.dps) for m in solr]]
    solr = solr[[mp.fabs(m) > 10**(-mp.mp.dps) for m in solr]]
    if Bound_nb == 1:
        solr = solr[[mp.im(m) < -10**(-mp.mp.dps) for m in solr]]
    if Bound_nb == 2:
        solr = solr[np.array([mp.im(m) < -10**(-mp.mp.dps) for m in solr]) +
                    np.array([mp.im(m) > 10**(-mp.mp.dps) for m in solr])]

    eigen1 = np.empty((len(solr), np.shape(M1)[0]), dtype=object)
    solr1 = np.array([], dtype=object)

    for i in range(len(solr)):
        M2 = mpmathM(M1.xreplace({kz: solr[i]}))
        global solrnow
        solrnow = solr[i]
        eigen1[i] = null_space(M2)

    existence = [not mp.isnan(sum(eigen1[i])) for i in range(len(solr))]
    # for soso in solr:
    #     print(soso)
    #     print('verif',(M1.xreplace({kz:soso}).det()).evalf())

    eigen1 = eigen1[existence]
    solr1 = solr[existence]
    # solr1 = solr1[[True,True,False]]
    # eigen1 = eigen1[[True,True,False]]

    if len(solr1) == 3:
        print("Inviscid semi infinite domain")
    elif len(solr1) == 6:
        print("Inviscid 2 boundaries")
    elif len(solr1) == 5:
        print("Viscous semi infinite domain")
    elif len(solr1) == 10:
        print("Viscous 2 boundaries")
    else:
        print("\033[1;31m" + "number of solution inconsistent," + '\033[1m' + '\033[0m', len(
            solr1), ",   total:", len(sol), "found")
    with mp.workdps(3):
        print('Kz      :      ', sympify(solr1))

    return(solr1, eigen1, M1)


def veigen(eig, sol):
    veig = 0
    for s in range(len(sol)):
        veig = veig + Symbol('C' + str(s)) * \
            eig[s] * ansatz.xreplace({kz: sol[s]})
    veig = veig
    return(veig)


def surfcond(val, dic):
    va = val.xreplace(dic).doit().xreplace(
        {f0.xreplace(dic): -(f - f0)}).xreplace(dic)
    return(va)


def surfcond_2(val, dic):
    va = val.xreplace(dic).doit().xreplace(
        {f0_2.xreplace(dic): -(f_2 - f0_2)}).xreplace(dic)

    return(va)


def Bound_nosolve(U, B, psi, psi_2b, dic, order_v, order_t):
    nn = n.xreplace(dic).doit()
    condB = params.condB
    condU = params.condU
    BC_list = []
###### Velocity conditions
    if condU == 'Stressfree':
        un = U.dot(nn)
        Eq_n1 = surfcond(un, dic).xreplace(dic)
        eu = strain(U) * nn.to_matrix(C)
        eu1 = eu.dot(tfox.to_matrix(C))
        eu2 = eu.dot(tfoy.to_matrix(C))
        Eq_BU1 = surfcond(eu1, dic)
        Eq_BU2 = surfcond(eu2, dic)
        if params.Bound_nb == 1:
            BC_list += [Eq_n1,Eq_BU1,Eq_BU2]
        elif params.Bound_nb == 2:
            nn2 = n2.xreplace(dic).doit()
            un2 = U.dot(nn2)
            Eq_n2 = surfcond_2(un2, dic)
            eu = strain(U) * nn2.to_matrix(C)
            eu1 = eu.dot(tfox2.to_matrix(C))
            eu2 = eu.dot(tfoy2.to_matrix(C))
            Eq2_BU1 = surfcond2(eu1, dic)
            Eq2_BU2 = surfcond2(eu2, dic)
            BC_list += [Eq_n2,Eq2_BU1,Eq2_BU2]
    elif condU == 'Inviscid':
        un = U.dot(nn)
        Eq_n1 = surfcond(un, dic).xreplace(dic)
        BC_list += [Eq_n1]
        if params.Bound_nb == 2:
            nn2 = n2.xreplace(dic).doit()
            un2 = U.dot(nn2)
            Eq_n2 = surfcond_2(un2, dic)
            BC_list += [Eq_n2]
###### Magnetic conditions
    if (condB == "harm pot"):
        bb = B + gradient(psi)
        Eq_b = surfcond(bb, dic)
        Eq_bx = Eq_b & C.i
        Eq_by = Eq_b & C.j
        Eq_bz = Eq_b & C.k
        BC_list += [Eq_bx,Eq_by,Eq_bz]
        if params.Bound_nb == 2:
            bb2 = B + gradient(psi_2b)
            Eq_b2 = surfcond_2(bb2, dic)
            Eq_b2x = Eq_b2 & C.i
            Eq_b2y = Eq_b2 & C.j
            Eq_b2z = Eq_b2 & C.k
            BC_list += [Eq_b2x,Eq_b2y,Eq_b2z]

    elif condB == "Thick":
        conservation_B = surfcond(B - psi, dic)
        if buf == 1:
            conservation_E = qRm * diff(B, z) - qRmm * diff(psi, z) + qAl * U
        else:
            conservation_E = nn.cross(
                qRm * curl(B) - qRmm * curl(psi) - (U.cross(B)))
        consE = surfcond(conservation_E, dic)
        Eq_Ex = consE & C.i
        Eq_Ey = consE & C.j
        Eq_Ez = consE & C.k
        Eq_bx = conservation_B & C.i
        Eq_by = conservation_B & C.j
        Eq_bz = conservation_B & C.k
        BC_list += [Eq_bx, Eq_by, Eq_bz, Eq_Ex, Eq_Ey]
        if params.Bound_nb == 2:
            conservation_B_2 = surfcond_2(B - psi_2b, dic)
            conservation_E_2 = nn2.cross(
                qRm * curl(B) - qRmc * curl(psi_2b) - (U.cross(B)))
            consE_2 = surfcond_2(conservation_E_2, dic)
            Eq_E2x = consE_2 & C.i
            Eq_E2y = consE_2 & C.j
            Eq_E2z = consE_2 & C.k
            Eq_b2x = conservation_B_2 & C.i
            Eq_b2y = conservation_B_2 & C.j
            Eq_b2z = conservation_B_2 & C.k
            BC_list += [Eq_b2x, Eq_b2y, Eq_b2z, Eq_E2x, Eq_E2y]


    if order_v == 'no':
        TEq = BC_list
    else:
        TEq = [taylor(eq, order_v, order_t, dic)
               for eq in BC_list]
    print('NUmber of BC: ',len(TEq))
    TEq = Matrix([powsimp(expand(tex.xreplace(dic)))
                  for tex in TEq])
    return(TEq)

#########################
###     Variables     ###
#########################


order = params.order
Bound_nb = params.Bound_nb
buf = params.buf
atmo = params.atmo
x = C.x
y = C.y
z = C.z


x.is_real
y.is_real
z.is_real

x._assumptions['real'] = True
y._assumptions['real'] = True
z._assumptions['real'] = True


t = Symbol('t', real=True)


# de = Del()
Dist, et, ev = symbols("Dist,et,ev", real=True)
Ri, Ro, Al, Rm, omega,  kx, ky, kxl, kyl, kz, rho_r, alpha, g, zeta, g0, g1, buoy, a, b, c, dv, fv, BOx, BOy, BOz, psi0, psi0_2b, p0 = symbols(
    "Ri, Ro, Al, Rm, omega, kx,ky,kxl,kyl,kz,rho_r,alpha, g,zeta, g0, g1, buoy, a, b, c, dv, fv,BOx, BOy, BOz,psi0,psi0_2b,p0")
U0, qRm, qRe, qRo, qFr, chi, qAl, qRmm, qRmc, Rl = symbols(
    "U0,qRm,qRe,qRo,qFr,chi,qAl,qRmm,qRmc,Rl", real=True)
bmx, bmy, bmz = symbols("bmx,bmy,bmz")
f0 = Function("f0")(x, y, z, t)
f1 = Function("f1")(x, y, z, t)
f2 = Function("f2")(x, y, z, t)
rho0 = Function("rho0")(x, y, z, t)
f0_2 = Function("f0_2")(x, y, z, t)
f1_2 = Function("f1_2")(x, y, z, t)
Ax = Function("Ax")(x, y, z, t)
Ay = Function("Ay")(x, y, z, t)
Az = Function("Az")(x, y, z, t)
Bx = Function("Bx")(x, y, z, t)
By = Function("By")(x, y, z, t)
Bz = Function("Bz")(x, y, z, t)
AgradB = ((Ax * D(Bx, x) + Ay * D(Bx, y) + Az * D(Bx, z)) * C.i +
          (Ax * D(By, x) + Ay * D(By, y) + Az * D(By, z)) * C.j
          + (Ax * D(Bz, x) + Ay * D(Bz, y) + Az * D(Bz, z)) * C.k)

b0x = Function("b0x")(x, y, z, t)
b0y = Function("b0y")(x, y, z, t)
b0z = Function("b0z")(x, y, z, t)
u0x = Function("u0x")(x, y, z, t)
u0y = Function("u0y")(x, y, z, t)
u0z = Function("u0z")(x, y, z, t)
f0 = Function("f0")(x, y, z, t)
f1 = Function("f1")(x, y, z, t)
f2 = Function("f2")(x, y, z, t)
rho0 = Function("rho0")(x, y, z, t)
f0_2 = Function("f0_2")(x, y, z, t)
f1_2 = Function("f1_2")(x, y, z, t)
f2_2 = Function("f2_2")(x, y, z, t)

u0 = Function("u0x")(x, y, z, t) * C.i + Function("u0y")(x, y,
                                                         z, t) * C.j + Function("u0z")(x, y, z, t) * C.k
b0 = Function("b0x")(x, y, z, t) * C.i + Function("b0y")(x, y,
                                                         z, t) * C.j + Function("b0z")(x, y, z, t) * C.k

ansatz = 0
g = -g0 * C.k
f = f0 + et * f1  # +et**2 *f2


# Vector normal and tangential to the topography
delf = gradient(f)
nfo = delf.normalize()
# nfo = C.k-gradient(-et*f1)
nx = (nfo & C.i)
ny = (nfo & C.j)
nz = (nfo & C.k)

if params.dom[f1] != 0:
    tfox = (C.i) + ((ny / nx) * C.j) + (-((nx**2 + ny**2) / (nx * nz)) * C.k)
    tfox = tfox.normalize()
    tfoy = (-(ny / nx) * C.i) + (C.j) + (0 * C.k)
    tfoy = tfoy.normalize()
else:
    tfox = (C.i)
    tfoy = (C.j)

if Bound_nb == 2:
    f_2 = f0_2 + Dist + et * f1_2
    delf2 = gradient(f_2)
    nfo2 = -delf2 / sqrt((delf2 & C.i)**2 + (delf2 & C.j)
                         ** 2 + (delf2 & C.k)**2)
    nx2 = (nfo2 & C.i)
    ny2 = (nfo2 & C.j)
    nz2 = (nfo2 & C.k)
    if params.dom[f1_2] != 0:
        tfox2 = (C.i) + ((ny2 / nx2) * C.j) + \
            (-((nx2**2 + ny2**2) / (nx2 * nz2)) * C.k)
        tfox2 = -tfox2 / sqrt((tfox2 & C.i)**2 +
                              (tfox2 & C.j)**2 + (tfox2 & C.k)**2)
        tfoy2 = (-(ny2 / nx2) * C.i) + (C.j) + (0 * C.k)
        tfoy2 = tfoy2 / sqrt((tfoy2 & C.i)**2 +
                             (tfoy2 & C.j)**2 + (tfoy2 & C.k)**2)
    else:
        tfox2 = (C.i)
        tfoy2 = (C.j)


######################
###     SCRIPT     ###
######################

# Parameters chosen for calculation
dico0 = params.dom
# dico1 = params.Buffet2010
dico1 = params.Glane
# dico1 = params.Me

LAT = params.LAT

# CHECK the realness of imposed Field
if params.test == 1:
    testre = Matrix([dico0[u0x], dico0[u0y], dico0[u0z], dico0[b0x],
                    dico0[b0y], dico0[b0z], dico0[f1], dico0[f1_2]])
    testre = np.imag(np.sum(np.array((testre.xreplace(
        {x: 10, y: 10, t: 10, omega: 10, ev: 1}).evalf()), dtype=np.complex128)))
    if testre > 1e-15:
        print('One of the parameters is not real')
        raise ValueError


if dico1[qRe] != 0 and params.condU == 'Inviscid':
    print("error Inviscid fluid incompatible with qRe != 0")
    raise(ValueError)

n = nfo.xreplace({**dico0, **dico1}
                 ).doit().xreplace({**dico0, **dico1})
# 1st tangential vector
tx = tfox.xreplace({**dico0, **dico1}
                   ).doit().xreplace({**dico0, **dico1})
# 2nd tangential vector
ty = tfoy.xreplace({**dico0, **dico1}
                   ).doit().xreplace({**dico0, **dico1})

# same for the 2nd boundary
if Bound_nb == 2:
    n2 = nfo2.xreplace({**dico0, **dico1}
                       ).doit().xreplace({**dico0, **dico1})

    tx2 = (tfox2.xreplace({**dico0, **dico1})
           ).doit().xreplace({**dico0, **dico1})
    ty2 = tfoy2.xreplace(
        {**dico0, **dico1}).doit().xreplace({**dico0, **dico1})

# Calculus for each order i
print('Utest', u0.xreplace({**dico0, **dico1}).coeff(ev))
test_e = u0.xreplace({**dico0, **dico1}).coeff(ev)
print(test_e)
if test_e == 0:
    ev_test = False
else:
    ev_test = True
comb = np.arange(order + 1)

if ev_test == True:
    list_1 = comb
    list_2 = comb
elif ev_test == False:
    list_1 = np.zeros(len(comb), dtype=np.int64)
    list_2 = comb

comb = np.unique(np.array(list(itertools.product(list_1, list_2))), axis=0)
aa = (np.sum(comb, axis=1)) * 10 + np.abs(comb[:, 1] - comb[:, 0])
idx = aa.argsort()
orders = comb[idx]
print(orders)
if ev_test == True:
    lim_part = 4
else:
    lim_part = 2

# ov = np.arange(order + 1)
# orders = list(itertools.product(ov, repeat=2))
# print(orders)
# ov = np.arange(order + 1)
# comb = list(comb(ov, repeat=2))

for i in range(0, len(orders)):
    iv = orders[i][0]
    it = orders[i][1]
    print("########################### ORDER", 'E', (iv, it))
    if atmo == 1:
        print("######## Atmospheric conditions ########")

    # with this I consider qAl is small
    if iv == 0 and it == 0:
        U = U0 * u0.xreplace({**dico0, **dico1})
        B = qAl * b0.xreplace({**dico0, **dico1})
        p = 0
        rho = rho0.xreplace({**dico0, **dico1})

        print('U0 : ', U)
        print('B0 : ', B)

        if params.condB == 'Thick':
            psi_2b = Symbol('psi0x_2b') * C.i + \
                Symbol('psi0y_2b') * C.j + Symbol('psi0z_2b') * C.k
            psi_2b = psi_2b + (Symbol('psi0x_2b_z') * C.i +
                               Symbol('psi0y_2b_z') * C.j + Symbol('psi0z_2b_z') * C.k) * z

            psi = Symbol('psi0x') * C.i + Symbol('psi0y') * \
                C.j + Symbol('psi0z') * C.k
            psi = psi + (Symbol('psi0x_z') * C.i + Symbol('psi0y_z')
                         * C.j + Symbol('psi0z_z') * C.k) * z

            if Bound_nb == 1:
                symb = [Symbol('psi0x'), Symbol('psi0y'), Symbol('psi0z'), Symbol(
                    'psi0x_z'), Symbol('psi0y_z'), Symbol('psi0z_z')]
            if Bound_nb == 2:
                symb = [Symbol('psi0x'), Symbol('psi0y'), Symbol('psi0z'), Symbol('psi0x_z'),
                        Symbol('psi0y_z'), Symbol('psi0z_z'), Symbol(
                            'psi0x_2b'), Symbol('psi0y_2b'),
                        Symbol('psi0z_2b_z'), Symbol('psi0x_2b_z'), Symbol('psi0y_2b_z'), Symbol('psi0z_2b_z')]

        if params.condB == 'harm pot':

            psi = Symbol('psi0') + Symbol('psi0_x') * x + \
                Symbol('psi0_y') * y + Symbol('psi0_z') * z
            psi_2b = Symbol('psi0_2b') + Symbol('psi0_x_2b') * \
                x + Symbol('psi0_y_2b') * y + Symbol('psi0_z_2b') * z

            if Bound_nb == 1:
                symb = [Symbol('psi0'), Symbol('psi0_x'),
                        Symbol('psi0_y'), Symbol('psi0_z')]
            if Bound_nb == 2:
                symb = [Symbol('psi0'), Symbol('psi0_x'), Symbol('psi0_y'), Symbol('psi0_z'),
                        Symbol('psi0_2b'), Symbol('psi0_x_2b'), Symbol('psi0_y_2b'), Symbol('psi0_z_2b')]

        eqpsi0 = Bound_nosolve(U, B, psi, psi_2b, {**dico0, **dico1}, iv, it)
        solpsi0 = solve(eqpsi0, symb, rational=False)

        psi = psi.xreplace(solpsi0)
        psi = psi.xreplace(dict(zip(symb, np.zeros(len(symb)))))
        psi_2b = psi_2b.xreplace(solpsi0)
        psi_2b = psi_2b.xreplace(dict(zip(symb, np.zeros(len(symb)))))

        so_part = zeros(8, 1)

        kzpsi = 0
        U_ord_0 = U
        B_ord_0 = B
        p_ord_0 = p
        rho_ord_0 = rho

        P0 = Symbol('P0_x') * x + Symbol('P0_y') * y + Symbol('P0_z') * z

        eqP = (makeEquations(U, B, P0, rho, 0, 0, dico0))

        p = P0.xreplace(solve(eqP, [Symbol('P0_x'), Symbol(
            'P0_y'), Symbol('P0_z')])).xreplace({**dico0, **dico1})

    elif iv == 1 and it == 0:
        P0 = Symbol('P0_x') * x + Symbol('P0_y') * y + Symbol('P0_z') * z
        p = p + ev * P0
        eqP = (makeEquations(U, B, p, rho, 1, 0, {**dico0, **dico1}))
        soeqp = solve(eqP, [Symbol('P0_x'), Symbol(
            'P0_y'), Symbol('P0_z')], rational=False)
        p = p.xreplace(soeqp)

        if params.condB == "Thick":
            TEq = Bound_nosolve(U, B, psi, psi_2b, {**dico0, **dico1}, iv, it)
            expo = findexp(TEq)
            B0 = B
            psi0 = psi
            for ex in expo:
                ansatz = ex * exp(I * kz * z)
                B0 += (Symbol('BOx') * C.i + Symbol('BOy') *
                       C.j + Symbol('BOz') * C.k) * ev * ansatz
                psi0 += (Symbol('bmx') * C.i + Symbol('bmy') *
                         C.j + Symbol('bmz') * C.k) * ev * ansatz

            eqpsi0 = diff(psi0, t) - (qRmm * laplacian(psi0))
            eqb0 = diff(B0, t) - (qRm * laplacian(B0) + (curl(U.cross(B0))))

            eqb0x = taylor(eqb0, 1, 0, {**dico0, **dico1})
            eqpsi0 = taylor(eqpsi0, 1, 0, {**dico0, **dico1})

            eqpsi0 = simp(eqpsi0)
            eqb0 = simp(eqb0)
            exp_time = findexp(Matrix([eqpsi0]))

            psi1 = psi
            B1 = B
            for ext in exp_time:
                equ = (simplify(comat(eqpsi0, ext) / ext)).to_matrix(C)
                sokzpsi = solve(equ[0], kz, rational=False)
                sokzpsi = np.array([m.evalf(mp.mp.dps) for m in sokzpsi])
                sokzpsi = sokzpsi[[mp.im(m.xreplace({**dico0, **dico1}).evalf(mp.mp.dps)) > 10**(
                    -mp.mp.dps) for m in sokzpsi]][0].xreplace({**dico0, **dico1}).evalf(mp.mp.dps)
                psi1 = psi1 + (Symbol('bmx') * C.i + Symbol('bmy')
                               * C.j) * ext.xreplace({kz: sokzpsi}) * ev

                equ = (simplify(comat(eqb0, ext) / ext)).to_matrix(C)
                sokzb = solve(equ[0], kz, rational=False)
                sokzb = np.array([m.evalf(mp.mp.dps) for m in sokzb])
                sokzb = sokzb[[mp.im(m.xreplace({**dico0, **dico1}).evalf(mp.mp.dps)) < -10**(-mp.mp.dps)
                               for m in sokzb]][0].xreplace({**dico0, **dico1}).evalf(mp.mp.dps)
                B1 = B1 + (Symbol('BOx') * C.i + Symbol('BOy') *
                           C.j) * ext.xreplace({kz: sokzb}) * ev

            psi_2b = 1

            Boun0 = Bound_nosolve(U, B1, psi1, psi_2b, {
                                  **dico0, **dico1}, 1, 0)

            exp_time = findexp(Matrix([B1.to_matrix(C), psi1.to_matrix(C)]))
            for ext in exp_time:

                def coe(x): return x.coeff(ext.xreplace({z: 0}))
                soB0 = solve(Boun0.applyfunc(coe),
                             dict=True, rational=False)[0]
                B += comat(B1, ext).xreplace(soB0)

                psi += comat(psi1, ext).xreplace(soB0)

    elif i >= lim_part:
        print('i=', i)
        print("Particular solution")
        eq = makeEquations(U, B, p, rho, iv, it, {**dico0, **dico1})
        var = [Symbol('u' + str(i) + 'x'), Symbol('u' + str(i) + 'y'), Symbol('u' + str(i) + 'z'), Symbol('p' + str(i)),
               Symbol('b' + str(i) + 'x'), Symbol('b' + str(i) + 'y'), Symbol('b' + str(i) + 'z'), Symbol('rho' + str(i))]
        Mc, rmec = linear_eq_to_matrix(eq, var)

        expo = findexp(rmec)

        so_part = zeros(8, 1)
        Up = U
        Bp = B
        pp = p
        rhop = rho

        for ex in expo:
            Up = Up + ev**iv * et**it * (Function('u' + str(i) + 'x')(t) * C.i + Function(
                'u' + str(i) + 'y')(t) * C.j + Function('u' + str(i) + 'z')(t) * C.k) * ex
            Bp = Bp + ev**iv * et**it * (Function('b' + str(i) + 'x')(t) * C.i + Function(
                'b' + str(i) + 'y')(t) * C.j + Function('b' + str(i) + 'z')(t) * C.k) * ex
            pp = pp + ev**iv * et**it * (Function('p' + str(i))(t)) * ex
            rhop = rhop + ev**iv * et**it * (Function('rho' + str(i))(t)) * ex

        varom = [Symbol('u' + str(i) + 'x'), Symbol('u' + str(i) + 'y'), Symbol('u' + str(i) + 'z'), Symbol('p' + str(i)),
                 Symbol('b' + str(i) + 'x'), Symbol('b' + str(i) + 'y'), Symbol('b' + str(i) + 'z'), Symbol('rho' + str(i))]

        var = Matrix([Function('u' + str(i) + 'x')(t), Function('u' + str(i) + 'y')(t), Function('u' + str(i) + 'z')(t),  Function('p' + str(i))(t),
                      Function('b' + str(i) + 'x')(t), Function('b' + str(i) + 'y')(t), Function('b' + str(i) + 'z')(t), Function('rho' + str(i))(t)])
        var_keep = var

        eqt = makeEquations(Up, Bp, pp, rhop, iv, it, {**dico0, **dico1})
        # expeq = findexp(eqt)
        #
        # eqtcount = eqt

        for ansatz in expo:
            var = var_keep

            def coe(x): return x.coeff(ansatz)
            eqp = eqt.applyfunc(coe)
            # eqtcount = eqtcount - eqp * ansatz
            print('ansatz', ansatz, '\n')

            # Tilt in the reference of the topography
            ansatz0 = ansatz.xreplace({z: 0})
            tilt = (simplify((gradient(ansatz0) / ansatz0)
                             ).normalize()).to_matrix(C)
            if sum(abs(tilt)) == 0:
                # tilt = ((C.i).normalize()).to_matrix(C)
                eqp.row_del(2)
            else:
                MNS = eqp.row((0, 1, 3))
                NS_modif = ((MNS.T) * (tilt))
                eqp[0] = NS_modif
                eqp.row_del(1)
            # print('eqp1',eqp,'\n')
            eqp = eqp.xreplace({y: 0})

            ### compute p ####
            # pop = solve(eqp,Symbol('p' + str(i)), rational=False)
            # print(pop)
            #
            # var = var.xreplace(pop)
            # eqp = eqp.xreplace(pop).doit()
            #
            Mp1, rmep1 = linear_eq_to_matrix(eqp.xreplace(
                dict(zip(var_keep, varom))).doit(), varom)
            # Mp1 = sympify(mpmathM(Mp1))
            # rmep1 = sympify(mpmathM(rmep1))
            # Mp1 = Mp1.evalf(mp.mp.dps)
            # rmep1 = rmep1.evalf(mp.mp.dps)

            # print('\n','Mp1',Mp1.evalf(3),'\n')
            # print('\n','rmep1',rmep1.evalf(3),'\n')
            # print('test Gauss jordan')
            # soluchap, pa = (Mp1.gauss_jordan_solve(rmep1))
            # print(soluchap,pa)

            # print('test LU')
            # soluchap = mp.lu_solve(Mp1, rmep1)
            # soluchap = Matrix(soluchap)
            # print(soluchap,pa)
            # try:
            #     soluchap, pa = (Mp1.gauss_jordan_solve(rmep1))
            #     print(pa)
            # except:
            #     try:
            #         print('Gauss pivot failed')
            #         soluchap = mp.lu_solve(Mp1, rmep1)
            #         soluchap = Matrix(soluchap)
            #         pa == Matrix(0, 1, [])
            #     except:
            #         try:
            #             print('Lu failed')
            #             print('test pinv')
            # try:
            #     for vv in varom:
            #         print('solve'+str(vv),solve(eqp.xreplace(
            #             dict(zip(var_keep, varom))).doit(),vv))
            #         print("solving without matrix")
            # except:
            #     print("didn't work")
            #     pass
            # try:
            #     soluchap, pa = (Mp1.gauss_jordan_solve(rmep1))
            # #     print(pa)
            #     soluchap = sympify(mp.chop(mpmathM(soluchap),tol = 10**(-mp.mp.dps/2)))
            # #     # print('gauss',soluchap.evalf(3))
            # #     # print('test gauss',(Mp1*soluchap-rmep1).evalf(3))
            # #
            # except:
                # print('Only one solution pinv', simplify(Mp1 * Mp1.pinv() * rmep1) == rmep1)
                # print('sol without arbitrary matrix', (Mp1.pinv_solve(rmep1)).evalf(3))

            # soluchap2 = Mp1.pinv_solve(rmep1)#, arbitrary_matrix=zeros(shape(Mp1)[0],1))
            # #soluchap2 = sympify(mp.chop(mpmathM(soluchap2),tol = 10**(-mp.mp.dps/2)))
            # print('sol pinv',soluchap2.evalf(3))
            # # print(soluchap2.evalf(2))
            # print('test pinv',simplify((Mp1*soluchap2-rmep1)).evalf(3))
            try:
                soluchap, pa = (Mp1.gauss_jordan_solve(rmep1))
                print('gauss succeed')
            except:
                soluchap = Mp1.pinv_solve(
                    rmep1, arbitrary_matrix=zeros(shape(Mp1)[0], 1))
                soluchap = sympify(
                    mp.chop(mpmathM(soluchap), tol=10**(-mp.mp.dps / 2)))
                print('sol pinv arbitrary matrix 0', soluchap.evalf(3))
                print('pinv succeed')
            # soluchap2 = Mp1.pinv_solve(rmep1, arbitrary_matrix=ones(shape(Mp1)[0],1)*1e5)
            # soluchap2 = sympify(mp.chop(mpmathM(soluchap2),tol = 10**(-mp.mp.dps/2)))
            # print('sol pinv arbitrary matrix 1e5',soluchap2.evalf(3))
            # #
            #
            # soluchap2 = Mp1.pinv_solve(rmep1, arbitrary_matrix=Matrix([0.5,1e3,-10,1e-3,5e5,9,8e2,3]))
            # soluchap2 = sympify(mp.chop(mpmathM(soluchap2),tol = 10**(-mp.mp.dps/2)))
            # print('sol pinv arbitrary matrix random',soluchap2.evalf(3))

            pa = Matrix(0, 1, [])

            #         except:
            #             print('pinv fail')
            #             pa = 1
            # print('soluchap',soluchap,pa)

            if pa == Matrix(0, 1, []):
                sop = soluchap * ansatz
                print('steady solution ok')
            else:
                raise(ValueError)
            # else:
            #     print('need time dependance')
            #
            #     # ##### Chop small values
            #     (A1, Aeq), Beq = linear_ode_to_matrix(eqp, var_keep, t, 1)
            #     # A1 = sympify(mpmathM(A1))
            #     # Aeq = sympify(mpmathM(Aeq))
            #     # print(Aeq)
            #     # Beq = sympify(mpmathM(Beq))
            #     # print(Beq)
            #     Aeq_k = Aeq
            #     Beq_k = Beq
            #     eqp = (A1 * Matrix(var_keep)).diff(t) - \
            #         Aeq * Matrix(var_keep) - Beq
            #     #
            #     # ######
            #
            #     eq_not = Matrix(np.array(eqp)[list(np.sum(A1, axis=1) == 0)])
            #     # bug of sympy solve
            #     eq_not = eq_not.row_insert(0, Matrix([100 * Symbol('ttu')]))
            #     eq_not = eq_not.row_insert(0, Matrix([100 * Symbol('ttu')]))
            #     ###############
            #     s_not = solve(eq_not, rational=False)[0]
            #     var = var.xreplace(s_not)
            #     eqp = eqp.xreplace(s_not).doit()
            #     ### compute p ####
            #     pop = solve(eqp, Function('p' + str(i))(t), rational=False)
            #     var = var.xreplace(pop)
            #     eqp = eqp.xreplace(pop).doit()
            #     (A1, Aeq), Beq = linear_ode_to_matrix(eqp, var_keep, t, 1)
            #
            #     eqs = Matrix(np.array(eqp)[list(np.sum(A1, axis=1) == 1)])
            #     var_t = Matrix(np.array(var_keep)[
            #                    list(np.sum(A1, axis=0) == 1)])
            #
            #     (A1, Aeq), Beq = linear_ode_to_matrix(eqs, var_t, t, 1)
            #
            #     AM = mpmathM(Aeq)
            #     BM = mpmathM(Beq)
            #     E, ER = mp.eig(AM)
            #     E = E
            #     ER = ER
            #
            #     hom = zeros(len(E), 1)
            #
            #     for kk in range(len(E)):
            #
            #         hom = hom + Symbol('C' + str(kk)) * \
            #             Matrix((ER[:, kk])) * exp(E[kk] * t)
            #     # try:
            #
            #     Aeq = sympify(mpmathM(Aeq))
            #     Beq = sympify(mpmathM(Beq))
            #     # part = mp.lu_solve(Aeq,-Beq)
            #
            #     print('\n', 'Aeq', Aeq.evalf(3), '\n')
            #     print('\n', 'Beq', Beq.evalf(3), '\n')
            #     try:
            #         part, pars = Aeq.gauss_jordan_solve(-Beq)
            #         print(pars)
            #         taus_zeroes = {tau: 0 for tau in pars}
            #         part_unique = part.xreplace(taus_zeroes)
            #
            #     except:
            #         print('Gauss pivot failed')
            #         print('test pinv')
            #         part = Aeq.pinv_solve(-Beq,
            #                               arbitrary_matrix=zeros(shape(Aeq)[0], 1))
            #         part_unique = part
            #
            #     # part,pars = soluchap,pa
            #         # part,pars = Aeq.gauss_jordan_solve(-Beq)
            #     # except:
            #     #     print("raté")
            #
            #     sol = hom + part_unique
            #     ics = solve(sol.xreplace({t: 0}), rational=False)
            #     soluchap = sol.xreplace(ics)
            #     soluchap = var.xreplace(dict(zip(var_t, soluchap)))
            #     sop = soluchap * ansatz
            so_part = so_part + sop

    #####################################
    ######  Homogeneous solution  #######
    #####################################
    if (iv, it) != (0, 0) and (iv, it) != (1, 0):
        print("Homogeneous solution")

        # Create the variable as Ubnd = Sum(U,0,n-1) + Upart + U hom
        Usopart = so_part[0] * C.i + \
            so_part[1] * C.j + so_part[2] * C.k
        Bsopart = so_part[4] * C.i + \
            so_part[5] * C.j + so_part[6] * C.k
        psopart = so_part[3]
        rhosopart = so_part[7]
        ### test of res of particular sol ###
        if params.test == 1:
            print('Test of the particular solutions at order E', (iv, it))
            testequations = makeEquations(
                U + ev**iv * et**it * Usopart, B + ev**iv * et**it * Bsopart, p + ev**iv * et**it * psopart, rho + ev**iv * et**it * rhosopart, iv, it, {**dico0, **dico1})
            testequations = simplify(testequations.xreplace(
                {x: 1, y: 0, z: -1, t: 0})).evalf()
            testequations = simplify(testequations.evalf())
            print('Residuals = ')
            print(testequations)
    #     #
        # + ev**iv * et**it * (Function('Uhomx')(x, y, z, t)
        Ubnd = U + ev**iv * et**it * Usopart
        # * C.i + Function('Uhomy')(x, y, z, t) * C.j + Function('Uhomz')(x, y, z, t) * C.k)
        # + ev**iv * et**it * (Function('Bhomx')(x, y, z, t)
        Bbnd = B + ev**iv * et**it * Bsopart
        # * C.i + Function('Bhomy')(x, y, z, t) * C.j + Function('Bhomz')(x, y, z, t) * C.k)
        # + ev**iv * et**it * (Symbol('Uhomx')* C.i + Symbol('Uhomy') * C.j + Symbol('Uhomz') * C.k)
        Ubnd = U + ev**iv * et**it * Usopart
        # + ev**iv * et**it *  (Symbol('Bhomx')* C.i + Symbol('Bhomy') * C.j + Symbol('Bhomz') * C.k)
        Bbnd = B + ev**iv * et**it * Bsopart

    # Calculate the boundary conditions equations
        psiBnd = psi
        psiBnd_2 = psi_2b

        TEq = Bound_nosolve(Ubnd, Bbnd, psiBnd, psiBnd_2,
                            {**dico0, **dico1}, iv, it)
        expo = findexp(TEq)
        # if iv == 1 and it == 1:
        #     expo = {exp(C.x*I + C.y*I + omega*I*t), exp(-C.x*I -C.y*I -omega*I*t)}
        # Uh = U_ord_0
        # Bh = B_ord_0
        # ph = p_ord_0
        # rhoh = rho_ord_0
        Uh = U + ev**iv * et**it * Usopart
        Bh = B + ev**iv * et**it * Bsopart
        ph = p + ev**iv * et**it * psopart
        rhoh = rho + ev**iv * et**it * rhosopart
        for ex in expo:
            ansatz = ex * exp(I * kz * z)

            Uh = Uh + ev**iv * et**it * (Symbol('u' + str(i) + 'x') * C.i + Symbol(
                'u' + str(i) + 'y') * C.j + Symbol('u' + str(i) + 'z') * C.k) * ansatz
            # if iv==1 and it ==0 and params.condB =='Thick':
            #     Bh = Bh + ev**iv * et**it * (Symbol('b' + str(i) + 'x') * C.i + Symbol(
            #         'b' + str(i) + 'y') * C.j ) * ansatz
            # else:
            Bh = Bh + ev**iv * et**it * (Symbol('b' + str(i) + 'x') * C.i + Symbol(
                'b' + str(i) + 'y') * C.j + Symbol('b' + str(i) + 'z') * C.k) * ansatz
            ph = ph + ev**iv * et**it * (Symbol('p' + str(i))) * ansatz
            rhoh = rhoh + ev**iv * et**it * (Symbol('rho' + str(i))) * ansatz

        # Calculate the governings equations

        # {**dico0, **dico1})
        eq = makeEquations(Uh, Bh, ph, rhoh, iv, it, {**dico0, **dico1})
        var = [Symbol('u' + str(i) + 'x'), Symbol('u' + str(i) + 'y'), Symbol('u' + str(i) + 'z'), Symbol('p' + str(i)),
               Symbol('b' + str(i) + 'x'), Symbol('b' + str(i) + 'y'), Symbol('b' + str(i) + 'z'), Symbol('rho' + str(i))]
        Mhtot, rmehtot = linear_eq_to_matrix(eq, var)

        expeq = findexp(Mhtot)

        if params.test == 1:
            print('rmehtot', simplify(simplify(rmehtot.xreplace({**dico0, **dico1}).xreplace(
                {x: 0, y: 0, t: 0, z: 0}).evalf())))

        UBnd = U + ev**iv * et**it * Usopart
        BBnd = B + ev**iv * et**it * Bsopart

        # Decompose equations by exponential create and solve matrix

        # expeq = list(expeq)
        sol = np.zeros(len(expeq), dtype=object)
        eig = np.zeros(len(expeq), dtype=object)
        kzpsi = np.zeros(len(expeq), dtype=object)
        kzpsi_2 = np.zeros(len(expeq), dtype=object)

        # Mhtotcount = Mhtot
        # print('expeq',expeq)

        for j, ansatz in enumerate(expeq):

            print('anstaz', ansatz)
            def coe(x): return x.coeff(ansatz)
            M = Mhtot.applyfunc(coe)
            ansatz0 = ansatz.xreplace({z: 0})
            tilt = (simplify((gradient(ansatz0) / ansatz0)
                             ).normalize()).to_matrix(C)
            print('tilt', tilt)
            if sum(abs(tilt)) == 0:
                print('NS not tilted')
                # tilt = ((C.j).normalize()).to_matrix(C)
                # MNS = M.row((0,1,3))
                # NS_modif = ((MNS.T)*(tilt)).T
                # M[0,:] = NS_modif
                M.row_del(2)  # remove useless equation
                M = M.xreplace({y: 0})
            else:

                MNS = M.row((0, 1, 3))
                NS_modif = ((MNS.T) * (tilt)).T
                M[0, :] = NS_modif
                M.row_del(1)  # remove useless equation
                M = M.xreplace({y: 0})

            sol[j], eig[j], M1 = eigen(M, {**dico0, **dico1})

            # Create variable necessary to apply boundary conditions

            UBnd = UBnd + ev**iv * et**it * (((Symbol('u' + str(i) + 'x') * C.i + Symbol(
                'u' + str(i) + 'y') * C.j + Symbol('u' + str(i) + 'z') * C.k))).xreplace(makedic(veigen(eig[j], sol[j]), i))
            BBnd = BBnd + ev**iv * et**it * (((Symbol('b' + str(i) + 'x') * C.i + Symbol(
                'b' + str(i) + 'y') * C.j + Symbol('b' + str(i) + 'z') * C.k))).xreplace(makedic(veigen(eig[j], sol[j]), i))

            psian = ev**iv * et**it * Symbol('psi' + str(i)) * ansatz
            if params.condB == "harm pot":
                kzpsi[j] = np.array(
                    solve(simplify(laplacian(psian) / psian), kz, rational=False))
                kzpsi[j] = kzpsi[j][[
                    np.imag(np.complex128(kps)) >= 0 for kps in kzpsi[j]]][0]
                psiBnd = psiBnd + ev**iv * et**it * \
                    (Symbol('psi' + str(i)) * ansatz).xreplace({kz: kzpsi[j]})
                if Bound_nb == 2:
                    kzpsi_2[j] = -kzpsi[j]
                    psiBnd_2 = psiBnd_2 + ev**iv * et**it * \
                        (Symbol('psi_2b' + str(i)) *
                         ansatz).xreplace({kz: kzpsi_2[j]})

            if params.condB == "Thick":

                if buf == 1:
                    kzpsi[j] = np.array(solve(simplify((diff(psian, t) - qRmm * diff(
                        psian, z, z)) / psian).xreplace({**dico1, **dico0}), kz, rational=False))
                else:
                    kzpsi[j] = np.array(solve(simplify((diff(psian, t) - qRmm * laplacian(
                        psian)) / psian).xreplace({**dico1, **dico0}), kz, rational=False))
                kzpsi[j] = np.array([kzps.evalf(mp.mp.dps)
                                    for kzps in kzpsi[j]])
                kzpsi[j] = kzpsi[j][[
                    np.imag(np.complex128(kps)) >= 0 for kps in kzpsi[j]]][0]
                psiBnd = psiBnd + ev**iv * et**it * \
                    ((Symbol('bmx') * C.i + Symbol('bmy') * C.j +
                     Symbol('bmz') * C.k) * ansatz).xreplace({kz: kzpsi[j]})

                if Bound_nb == 2:
                    kzpsi_2[j] = np.array(solve(simplify((diff(psian, t) - qRmc * laplacian(
                        psian)) / psian).xreplace({**dico1, **dico0}), kz, rational=False))
                    kzpsi_2[j] = np.array([kzps.evalf(mp.mp.dps)
                                          for kzps in kzpsi_2[j]])
                    kzpsi_2[j] = kzpsi_2[j][[
                        np.imag(np.complex128(kps)) <= 0 for kps in kzpsi_2[j]]][0]
                    psiBnd_2 = psiBnd_2 + ev**iv * et**it * \
                        ((Symbol('bmx_2') * C.i + Symbol('bmy_2') * C.j +
                         Symbol('bmz_2') * C.k) * ansatz).xreplace({kz: kzpsi_2[j]})
        # Calculate boundary conditions equations
        # if params.test == 1:
        #     print('Mhtotcount', simplify(simplify(Mhtotcount)))

        # Add constant value for homogeneous solutions
        UBnd += ev**iv * et**it * (Symbol('CUx') * C.i + Symbol('CUy')
                                   * C.j + Symbol('CUz') * C.k)
        BBnd += ev**iv * et**it * (Symbol('CBx') * C.i + Symbol('CBy')
                                   * C.j + Symbol('CBz') * C.k)
        Eqbound = Bound_nosolve(UBnd, BBnd, psiBnd, psiBnd_2, {
            **dico0, **dico1}, iv, it)
    # Decompose equations by exponential create and solve matrix

        if params.condB == "Thick":
            if Bound_nb == 1:
                coeffs = [Symbol('C' + str(i)) for i in range(len(Eqbound) - 3)
                          ] + [Symbol('bmx'), Symbol('bmy'), Symbol('bmz')]
            if Bound_nb == 2:
                coeffs = [Symbol('C' + str(i)) for i in range(len(Eqbound) - 6)] + [Symbol('bmx_2'),
                                                                                    Symbol('bmy_2'), Symbol('bmz_2')] + [Symbol('bmx'), Symbol('bmy'), Symbol('bmz')]

        else:
            if Bound_nb == 1:
                coeffs = [Symbol('C' + str(i)) for i in range(len(Eqbound) - 1)] + [Symbol(
                    "psi" + str(i))]
            if Bound_nb == 2:
                coeffs = [Symbol('C' + str(i)) for i in range(len(Eqbound) - 2)] + [Symbol(
                    "psi" + str(i)), Symbol("psi_2b" + str(i))]
        Matbtot, resbtot = linear_eq_to_matrix(Eqbound, coeffs)

        Matcount = Matbtot
        rescount = resbtot
        for j, ansatz in enumerate(expeq):
            print('ansatz', ansatz.evalf(3))

            def coe(x): return x.coeff(ansatz.xreplace({z: 0}))

            Mat = (Matbtot).applyfunc(coe)  # .evalf(mp.mp.dps*3)
            res = (resbtot).applyfunc(coe)  # .evalf(mp.mp.dps*3)

            # try:
            #     abc,par_abc = Mat.gauss_jordan_solve(res)
            # except:
            #     print('Gauss pivot failed')
            try:
                abc, par_abc = Mat.gauss_jordan_solve(res)
            except:
                print('Gauss pivot failed')
                print('pinv')

                abc = Mat.pinv_solve(
                    res, arbitrary_matrix=zeros(shape(Mat)[0], 1))

            # verification
            Matcount = Matcount - ((Matbtot).applyfunc(coe)
                                   * (ansatz.xreplace({z: 0})))
            rescount = rescount - ((rescount).applyfunc(coe)
                                   * (ansatz.xreplace({z: 0})))
            # if abs(sum(((Matbtot).applyfunc(coe) * Matrix(abc) - (resbtot).applyfunc(coe)).evalf())) > 10**(-mp.mp.dps+mp.mp.dps/5) :
            #     print('error of :', abs(sum(((Matbtot).applyfunc(coe) *
            #                                  Matrix(abc) - (resbtot).applyfunc(coe)).evalf())))
            #     raise ValueError

            # Write homogeneous solution
            solhom = zeros(8, 1)

            for l in range(len(sol[j])):

                solhom = solhom + \
                    abc[l] * Matrix(eig[j][l]) * \
                    ansatz.xreplace({kz: sol[j][l]})

            abc = list(abc)
    # Expression of the final calculated variable Un

            U = U + ev**iv * et**it * (solhom[0] * C.i + solhom[1]
                                       * C.j + solhom[2] * C.k)
            B = B + ev**iv * et**it * (solhom[4] * C.i + solhom[5]
                                       * C.j + solhom[6] * C.k)
            p = p + ev**iv * et**it * solhom[3]
            rho = rho + ev**iv * et**it * solhom[7]

            if params.condB == "Thick":
                psi = psi + ev**iv * et**it * (abc[-3] * C.i + abc[-2] * C.j + abc[-1] * C.k) * \
                    ansatz.xreplace({kz: kzpsi[j]})
                if Bound_nb == 2:
                    psi_2b = psi_2b + ev**iv * et**it * (abc[-6] * C.i + abc[-5] * C.j + abc[-4] * C.k) * \
                        ansatz.xreplace({kz: kzpsi_2[j]})

            else:
                if Bound_nb == 1:
                    psi = psi + ev**iv * et**it * abc[-1] * \
                        ansatz.xreplace({kz: kzpsi[j]})
                if Bound_nb == 2:
                    psi = psi + ev**iv * et**it * abc[-2] * \
                        ansatz.xreplace({kz: kzpsi[j]})
                    psi_2b = psi_2b + ev**iv * et**it * \
                        abc[-1] * ansatz.xreplace({kz: -kzpsi[j]})

        rescount = simplify(simplify(simplify(rescount))).xreplace({x: 1})
        if params.test == 1:
            print('Matcount', simplify(simplify(simplify(Matcount))))
            print('rescount ::::::::::::::::::::', rescount)

        U = U + ev**iv * et**it * Usopart
        B = B + ev**iv * et**it * Bsopart
        p = p + ev**iv * et**it * psopart
        rho = rho + ev**iv * et**it * rhosopart
        # B = B + ev**iv * et**it*(rescount[1]*C.i+rescount[2]*C.j+rescount[3]*C.k)

        const = solve(rescount, rational=False)
        print('const', const)
        U += ev**iv * et**it * (Symbol('CUz') * C.k)
        B += ev**iv * et**it * (Symbol('CBx') * C.i + Symbol('CBy')
                                * C.j + Symbol('CBz') * C.k)
        U = U.xreplace(const)
        B = B.xreplace(const)

        U = (U.xreplace({**dico0, **dico1}).xreplace({**dico0, **dico1})
             ).xreplace({Symbol('tau0'): 0})  # .xreplace({Symbol('tau1'):1})
        B = (B.xreplace({**dico0, **dico1}).xreplace({**dico0, **dico1})
             ).xreplace({Symbol('tau0'): 0})  # .xreplace({Symbol('tau1'):1})
        # .xreplace({Symbol('tau0'):1}).xreplace({Symbol('tau1'):1})
        rho = (rho.xreplace({**dico0, **dico1}).xreplace({**
               dico0, **dico1})).xreplace({Symbol('tau0'): 0})
        # .xreplace({Symbol('tau0'):1}).xreplace({Symbol('tau1'):1})
        p = (p.xreplace({**dico0, **dico1}).xreplace({**
             dico0, **dico1})).xreplace({Symbol('tau0'): 0})
        # psi = psi.xreplace({Symbol('tau0'):0})#.xreplace({Symbol('tau0'):1}).xreplace({Symbol('tau1'):1})
        # psi_2b = psi_2b.xreplace({Symbol('tau0'):0})#.xreplace({Symbol('tau0'):1}).xreplace({Symbol('tau1'):1})

        # print('\n U : ',simp(U.evalf(3)),'\n')
        # print('\n B : ',simp(B.evalf(3)),'\n')
        # print('\n p : ',simp(p.evalf(3)),'\n')
        # print('\n rho : ',simp(rho.evalf(3)),'\n')
        # print('\n psi : ',simp(psi.evalf(3)),'\n')
        #
    test_res = False
    if test_res == True:
        testbound = Bound_nosolve(U, B, psi, psi_2b, {
            **dico0, **dico1}, 'no', 'no')
        print('testbound0K')

        zeta_test = 1e-5
        lim_res = 1e-12
        res_test = 1
        while lim_res * (2) < res_test or res_test < lim_res / (2):
            sumres = 0
            for j, ree in enumerate(testbound):
                free = lambdify([x, ev, et], ree**2)
                sumres += integ.quad(lambda x: free(x, 1,
                                     zeta_test), -np.pi, np.pi)[0] / (2 * np.pi)
            res_test = sumres
            if res_test > lim_res:
                zeta_test = zeta_test / 2
            elif res_test < lim_res:
                    zeta_test = zeta_test * 1.5
            print(res_test, zeta_test)
        print('final', res_test, zeta_test)
    ### TEST ###
    if params.test == 1:

        testbound = Bound_nosolve(U, B, psi, psi_2b, {
            **dico0, **dico1}, iv, it)
        testbound = testbound.xreplace({x: mp.pi / 4, y: 0, t: 0}).evalf()
        # print('testbound',testbound)
        testbound = mp.chop(mpmathM(testbound.evalf(3)),
                            tol=10**(-mp.mp.dps / 1.2))
        print('BC Order' + str(iv) + ',' + str(it))
        print(testbound)

        testequations = makeEquations(U, B, p, rho, iv, it, {**dico0, **dico1})
        testequations = testequations.xreplace(
            {x: mp.pi / 4, y: 0, t: 0, z: 0}).evalf()
        # print('testequations',testequations)
        testequations = mp.chop(
            mpmathM(testequations.evalf(3)), tol=10**(-mp.mp.dps / 1.2))
        print('Equations Order ' + str(iv) + ',' + str(it))
        print(testequations)

        #
        # print('All Bound Cond' + str(iv)+','+str(it))
        # un = U.dot(n)
        # Eq_n1 = surfcond(un, {**dico0, **dico1}).xreplace({**dico0, **dico1})
        # conservation_B = surfcond(B-psi,{**dico0, **dico1})
        # conservation_E = qRm*curl(B)-qRmm*curl(psi)-(U.cross(B))
        # if buf==1:
        #     conservation_E = qRm*diff(B,z)-qRmm*diff(psi,z)+qAl*U
        # consE = surfcond(conservation_E,{**dico0, **dico1})
        # Mat= [Eq_n1,consE&C.i,consE&C.j,consE&C.k,conservation_B&C.i,conservation_B&C.j,conservation_B&C.k]
        # Mat = Matrix([taylor(eq,iv,it,{**dico0, **dico1})
        #        for eq in Mat])
        # Mat =Mat.xreplace({x: mp.pi/4, y: 0, t: 0, z: 0}).xreplace({et:1,ev:1}).evalf()
        # testequations = mpmathM(Mat.evalf(3))
        # print('All BC Order ' + str(iv)+','+str(it))
        # print(testequations)

        # test of the residuals of the boundary conditions:

            # zeta0 = 1e-2
            # res = 1
            # lim_res =1e-5
            # while res > lim_res:
            #     res = np.abs(np.real(np.complex128(sum(testbound.xreplace({et:zeta0}).evalf()))))
            #     zeta0= zeta0/2
            #     print(zeta0)
            # print('zeta0 min =',zeta0,'  with res ', res)
            # print(testbound.xreplace({et:zeta0}).evalf())


print("Finish")
########################
###   Post Process   ###
########################

if params.pressure_stress == True:
    print('Pressure Stress....')
    Bt = simp(realve(B))
    P = simp(((conj(p) + p) / 2))
    nn = simp(realve(n))
    Fp = surfcond(simp((P * nn)), {**dico0, **dico1})
    Fpm = surfcond(simp((((Bt.dot(Bt)) / 2) * nn)), {**dico0, **dico1})
    Fptay = taylor_serie((Fp & C.i) - (Fpm & C.i), iv * 2, it * 2, {})
    ValPress = meanterm(expand((Fptay)))
    print('= ', ValPress.evalf())
    QFR = params.QFR
    print([QFR, str(re(ValPress.evalf()))])

    # print('test previous method')
    #
    # p = (p + conjugate(p)) / 2
    # nz = (n & C.i)
    # nre = ((nz + conjugate(nz)) / 2)
    #
    #
    # Fp = ((p * nre)).xreplace({conjugate(x): x,
    #                            conjugate(y): y, conjugate(z): z})
    # Fptay = surfcond(Fp, {**dico0, **dico1}).xreplace({et: 1e-3, p0: 0,t:0})
    # #Fptay = taylor_series(Fp, 0, (order * 2),e).xreplace({e: zeta, p0: 0,t:0}).xreplace(dic)
    # FFp = lambdify((x, y), Fptay, 'mpmath')
    # print("precessing quad")
    # avFp = mp.quad(FFp, [-mp.pi, mp.pi], [-mp.pi, mp.pi], maxdegree=5,verbose = True,method='gauss-legendre')
    # avFp = avFp / (4 * mp.pi**2)
    # print(avFp)

if params.ohmic_dissipation == True:
    print('Ohmic Dissipation....')
    Bt = simp(realve(B))
    if params.condB == 'Thick':
        psit = simp(realve(psi))
    if params.condB == 'harm pot':
        psit = 0 * C.i  # bof
    cuB = curl(Bt)
    cupsi = curl(psit)
    Phi = qRm * ((cuB).dot(cuB))  # - (cuB).dot(U.cross(B))
    Phi_m = qRmm * ((cupsi).dot(cupsi))
    Phi_T = taylor_serie(Phi, iv * 2, it * 2, {})
    Phi_m_T = taylor_serie(Phi_m, iv * 2, it * 2, {})
    Phi_T = meanterm(Phi_T)
    Phi_m_T = meanterm(Phi_m_T)
    Diss = simp(((integrate(Phi_T, (z, -oo, 0)) + integrate(Phi_m_T,
                (z, 0, oo))).xreplace({**dico0, **dico1}))).evalf()
    # print('Dissipation at order',OO[1],OO[0],' = ',Diss.evalf())
    QFR = params.QFR
    print([QFR, str(re(Diss.evalf()))])
    # ov = np.arange((order*2)+1)
    # orders = list(comb(ov,repeat=2))
    # orders = [[2,2]]
    # for OO in orders:
    #     try:
    #         Phi_T = taylor(Phi,OO[1],OO[0],{})
    #         Phi_m_T = taylor(Phi_m,OO[1],OO[0],{})
    #         Phi_T = meanterm(Phi_T)
    #         Phi_m_T = meanterm(Phi_m_T)
    #         Diss = simp(((integrate(Phi_T,(z,-oo,0))+integrate(Phi_m_T,(z,0,oo))).xreplace({**dico0, **dico1}))).evalf()
    #         print('Dissipation at order',OO[1],OO[0],' = ',Diss.evalf())
    #         print([LAT,np.real(np.complex128(Diss.evalf()))])
    #     except:
    #         print('Dissipation at order',OO[1],OO[0],' = 0...')


if params.write_file == True:
    topo1 = (-(f - f0)).xreplace({**dico0, **dico1})
    dic_meta = {'U0x': U0 * dico0[u0x],
                'U0y': U0 * dico0[u0y],
                'B0x': qAl * dico0[b0x],
                'B0y': qAl * dico0[b0y],
                'B0z': qAl * dico0[b0z],
                'Ro': 1 / dico1[qRo],
                'LAT': LAT,
                'Al': 1 / dico1[qAl],
                'Fr': 1 / dico1[qFr],
                'Rm': 1 / dico1[qRm],
                'Rmm': 1 / dico1[qRmm],
                'Rmc': 1 / dico1[qRmc],
                'Rl':  dico1[Rl],
                'omega': dico1[omega],
                'condB': params.condB,
                'atmo': params.atmo,
                'order_list': orders.tolist()}
    # for keys in dic_meta:
    #     try:
    #         dic_meta[keys] = str(S(dic_meta[keys]).evalf(3))
    #     except:
    #         pass
    data = {
        'topo': topo1,
        'meta': dic_meta,
        'Expr': Matrix([U & C.i, U & C.j, U & C.k, p,
                    B & C.i, B & C.j, B & C.k, rho, psi])
                    }
    if Bound_nb == 2:
        topo2 = (-(f_2 - f0_2)).xreplace({**dico0, **dico1})
        data['topo2'] = topo2
    with open(params.filename + '.dat', "wb") as f:
        pickle.dump(data, f)
