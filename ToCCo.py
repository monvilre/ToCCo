from sympy import *
from sympy import E, Eq, Function, pde_separate, Derivative as D, Q
from sympy.vector import CoordSys3D,matrix_to_vector,curl,gradient,divergence,Del,Divergence,Gradient, laplacian,Curl
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols
from sympy.physics.quantum import TensorProduct
from sympy.solvers.solveset import linsolve
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from sympy.matrices import matrix_multiply_elementwise
import numpy as np
import mpmath as mp
import params
mp.mp.dps = params.prec

C = CoordSys3D('C')

#######################
###    FUNCTIONS    ###
#######################
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)

# Taylor approximation at x0 of the function 'function'
def taylor_series(function, x0, n, x ):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x, i).xreplace({x: x0}))/(factorial(i))*(x - x0)**i
        i += 1
    return p

# Convert Sympy Matrix to mpmath array
def mpmathM(A):
    B = mp.matrix(A.shape[0],A.shape[1])
    for k in range(A.shape[0]):
            for l in range(A.shape[1]):
                B[k,l] = mp.mpc(str(re(A[k,l])),str(im(A[k,l])))
    return(B)

#Find the null space of the mpmath matrix A
def null_space(A, rcond=None):
    u, s, vh = mp.svd_c(A)

    M, N = u.rows, vh.cols
    if rcond is None:
        rcond = 10**(-(mp.mp.dps)) * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum([x > tol for x in s], dtype=int)
    Q = vh[num:,:].transpose_conj()

    lo = Q.cols
    pa=0
    while lo != 1:
        if lo > 1:
            rcond = rcond/2
        if lo ==0:
            rcond = rcond*10
        u, s, vh = mp.svd_c(A)
        M, N = u.rows, vh.cols
        tol = np.amax(s) * rcond
        num = np.sum([x > tol for x in s], dtype=int)
        Q = vh[num:,:].transpose_conj()
        lo = Q.cols
        pa = 1
    if pa ==1 :
        print("precision issue, warning, rcond = ",rcond)

    return(Q)

#Return the Taylor term of exp at order n
def taylor(exp,n,dic):
    expt = ((exp.xreplace(dic).doit()).taylor_term(n,e)/e**n)
    return expt

#Create the Matrix from MHD equation
def makeMatrix(U,B,p,order,dic,vort =True):
    # if order !=0:

    buoy = Ri*(rho*g)
    # else:
    #     r1 = rho0
    #     buoy = Ri*r1*g
    Cor=((2+chi*y)*qRo*C.k).cross(U)
    BgradB = (AgradB.xreplace({Ax:B&C.i,Ay:B&C.j,Az:B&C.k,Bx:B&C.i,By:B&C.j,Bz:B&C.k})).doit()
    UgradU = (AgradB.xreplace({Ax:U&C.i,Ay:U&C.j,Az:U&C.k,Bx:U&C.i,By:U&C.j,Bz:U&C.k})).doit()
    Eq_NS = diff(U,t)+Cor+UgradU-(-gradient(p)+qRe*laplacian(U)+buoy+BgradB)
    Eq_vort=diff((Eq_NS&C.j),x)-diff((Eq_NS&C.i),y)
    Eq_m=divergence(U)
    Eq_b=diff(B,t)- (qRm*laplacian(B) + curl(U.cross(B)))

    if vort == True:
        eq = zeros(7,1)
        for i,j in enumerate([Eq_NS&C.i,Eq_vort,Eq_NS&C.k,Eq_b&C.i,Eq_b&C.j,Eq_b&C.k,Eq_m]):
            eq[i] = taylor(j,order,dic)
        var = [Symbol('u'+str(order)+'x'),Symbol('u'+str(order)+'y'),Symbol('u'+str(order)+'z'),Symbol('p'+str(order)),Symbol('b'+str(order)+'x'),Symbol('b'+str(order)+'y'),Symbol('b'+str(order)+'z')]
        M, rme = linear_eq_to_matrix(eq, var)
        M = simplify((M/ansatz)).xreplace(dic)

        print("Matrix OK")

    return(M,rme,r1)

# Return the solution vectors corresponding to the matrix of the problem
def eigen(M,dic,order):
    M1 = M.xreplace(dic)
    with mp.workdps(int(mp.mp.dps*2)):
        M1 = M1.evalf(mp.mp.dps)
        det =(M1).det(method = 'berkowitz')
        detp = Poly(det,kz)
        co = detp.all_coeffs()
        co = [mp.mpc(str(re(k)),str(im(k))) for k in co]

        maxsteps = 3000
        extraprec = 500
        ok =0
        while ok == 0:
            try:
                sol,err = mp.polyroots(co,maxsteps =maxsteps,extraprec = extraprec,error =True)
                sol = np.array(sol)
                print("Error on polyroots =", err)
                ok=1
            except:
                maxsteps = int(maxsteps*2)
                extraprec = int(extraprec*1.5)
                print("Poly roots fail precision increased: ",maxsteps,extraprec)
    te = np.array([mp.fabs(m) < mp.mpf(10**mp.mp.dps) for m in sol])
    solr = sol[te]

    if Bound_nb == 1:
        solr = solr[[mp.im(m) < 0 for m in solr]]
    eigen1 = np.empty((len(solr),np.shape(M1)[0]),dtype = object)

    with mp.workdps(int(mp.mp.dps*2)):
        for i in range(len(solr)):
            M2 = mpmathM(M1.xreplace({kz:solr[i]}))
            eigen1[i] = null_space(M2)
        solr1 = solr

        div = [mp.fabs(x) for x in (order*kxl.xreplace(dic)*eigen1[:,4]+order*kyl.xreplace(dic)*eigen1[:,5]+solr1*eigen1[:,6])]
        testdivB = [mp.almosteq(x,0,10**(-(mp.mp.dps/2))) for x in div]
        eigen1 =eigen1[testdivB]
        solr1 = solr1[testdivB]
        if len(solr1) == 3:
            print("Inviscid semi infinite domain")
        elif len(solr1) == 6:
            print("Inviscid 2 boundaries")
        elif len(solr1) == 5:
            print("Viscous semi infinite domain")
        elif len(solr1) == 10:
            print("Viscous 2 boundaries")
        else:
            print("number of solution inconsistent,",len(solr1))

    return(solr1,eigen1,M1)


def makedic(eig,order):
    dic = {
        Symbol('u'+str(order)+'x'):eig[0],
        Symbol('u'+str(order)+'y'):eig[1],
        Symbol('u'+str(order)+'z'):eig[2],
        Symbol('p'+str(order)):eig[3],
        Symbol('b'+str(order)+'x'):eig[4],
        Symbol('b'+str(order)+'y'):eig[5],
        Symbol('b'+str(order)+'z'):eig[6],
    }
    return(dic)

def veigen(eig,sol):
    veig=0
    for s in range(len(sol)):
        veig = veig + Symbol('C'+str(s))*eig[s]*ansatz.xreplace({kz:sol[s]})
    veig = veig/ansatz

    return(veig)

#Surface condition of the 1st boundary
def surfcond(val,dic,realtopo =True ):
    if realtopo==True:
        va = val.xreplace(dic).doit().xreplace({kz:mp.sqrt(-(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dic)),f0.xreplace(dic):-(f-f0)}).xreplace(dic)
    else:
        va = val.xreplace(dic).doit().xreplace({kz:mp.sqrt(-(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dic)),f0.xreplace(dic):(-(f-f0)-conjugate(f-f0))/2}).xreplace(dic)
    return(va)

#Surface condition of the 2nd boundary
def surfcond_2(val,dic,realtopo =True ):
    if realtopo==True:
        va = val.xreplace(dic).doit().xreplace({kz:-mp.sqrt(-(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dic)),f0_2.xreplace(dic):-(f_2-f0_2)}).xreplace(dic)
    else:
        va = val.xreplace(dic).doit().xreplace({kz:-mp.sqrt(-(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dic)),f0_2.xreplace(dic):(-(f_2-f0_2)-conjugate(f_2-f0_2))/2}).xreplace(dic)
    return(va)

#Calculate tangential pressure stress
def pressure(p,n,dic,order):
    p= (p+conjugate(p))/2
    nz = (n&C.i)
    nre = ((nz+conjugate(nz))/2)
    nre = (series(nz,e,0,order+1).removeO())
    Fp = ((p*nre)).xreplace({conjugate(x):x,conjugate(y):y,conjugate(z):z})
    #~ Fp = surfcond(Fp,dic,realtopo=True)
    Fp = Fp.xreplace({z:topo_sum})
    Fptay = (taylor_series(Fp,0,(order*2),e)).xreplace({e:zeta}).xreplace(dic)
    # Sympy implemented series
    #~ Fptay = (series(Fp,e,0,(order*2+1)).removeO()).xreplace({e:zeta}).xreplace(dic)
    FFp = lambdify((x,y),Fptay,'mpmath')
    avFp= mp.quad(FFp, [-mp.pi, mp.pi],[-mp.pi, mp.pi],maxdegree =10)
    avFp = avFp/(4*mp.pi**2)
    return(avFp)

#Calculate the strain tensor
def strain(U):
    xy = 1/2*(diff(U&C.i,y)+diff(U&C.j,x))
    xz = 1/2*(diff(U&C.i,z)+diff(U&C.k,x))
    yz = 1/2*(diff(U&C.j,z)+diff(U&C.k,y))
    strainT = Matrix([[diff(U&C.i,x),xy,xz],
                      [xy,diff(U&C.j,y),yz],
                      [xz,yz,diff(U&C.k,z)]])
    return(strainT)

def Bound(U,B,sol,eig,dic,order,condB = "harm pot",condU = 'Inviscid'):
    lso = len(sol)
    for s in range(len(sol)):
        globals() ['C'+str(s)] = Symbol('C'+str(s))
    #################################
    ###   Inviscid solution :     ###
    ###  lso=3 --> 1 boundary     ###
    ###  lso=6 --> 2 boundaries   ###
    #################################
    nn = n.xreplace(dic).doit()
    U = (U.xreplace(makedic(veigen(eig,sol),order))).xreplace({U0:1})
    B = (B.xreplace(makedic(veigen(eig,sol),order)))
    if (condB == "harm pot"):
            bchx,bchy,bchz = symbols("bchx,bchy,bchz")
            bcc =surfcond((bchx*C.i +bchy*C.j + bchz*C.k)*ansatz - gradient(psi),dic).doit()
            sob = list(linsolve([bcc&C.i,bcc&C.j,bcc&C.k],(bchx,bchy,bchz)))[0]
            bbc = sob[0]*C.i + sob[1]*C.j + sob[2]*C.k
            bb = B.xreplace(makedic(veigen(eig,sol),order)) - (bbc*ansatz)
            Eq_b= surfcond(bb,dic)
            Eq_bx = Eq_b&C.i; Eq_by = Eq_b&C.j; Eq_bz = Eq_b&C.k
            if params.Bound_nb ==2:
                bchx2,bchy2,bchz2 = symbols("bchx2,bchy2,bchz2")
                bcc2 =surfcond_2((bchx2*C.i +bchy2*C.j +bchz2*C.k)*ansatz - gradient(psi_2b),dic)
                sob2 = list(linsolve([bcc2&C.i,bcc2&C.j,bcc2&C.k],(bchx2,bchy2,bchz2)))[0]
                bbc2 = sob2[0]*C.i + sob2[1]*C.j + sob2[2]*C.k
                bb2 = B.xreplace(makedic(veigen(eig,sol),order)) - (bbc2*ansatz)
                Eq_b2= surfcond_2(bb2,dic)
                Eq_b2x = Eq_b2&C.i; Eq_b2y = Eq_b2&C.j; Eq_b2z = Eq_b2&C.k
    if (condB == "thick"):
            bchx,bchy,bchz,eta = symbols("bchx,bchy,bchz,eta")
            kz_t = -sqrt(-kxl**2-kyl**2-I*omega/qRmm)
            # kz_t = I*1e12
            B_mant = (bchx*C.i +bchy*C.j + bchz*C.k)*ansatz.xreplace({kz:kz_t})


            eq_ind = (surfcond((diff(B_mant,t)-qRmm*laplacian(B_mant)-diff(B,t) +qRm*laplacian(B) - curl(U.cross(B))).xreplace({kz:kz_t}),dic).xreplace(dic))
            eq_E = (qRm*curl(B)-U.cross(B)-qRmm*curl(B_mant))
            eq_Et = surfcond(((nn).cross(eq_E)).xreplace({kz:kz_t}),dic)




            eq_B = surfcond(((B_mant.dot(nn)-B.dot(nn))).xreplace({kz:kz_t}),dic)

            un = (U.dot(nn))
            Eq_n1= surfcond((un).xreplace({kz:kz_t}),dic).xreplace(dic)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,eq_ind&C.i,eq_ind&C.j,eq_ind&C.k,eq_Et.dot(tx),eq_Et.dot(ty)]]

            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,bchx,bchy,bchz))




    U = (U.xreplace(makedic(veigen(eig,sol),order))).xreplace({U0:1})
    un = U.dot(nn)
    Eq_n1= surfcond(un,dic).xreplace(dic)

    if condU == "Inviscid":
        if params.Bound_nb ==2:
            nn2 = n2.xreplace(dic).doit()
            un2 = U.dot(nn2)
            Eq_n2= surfcond_2(un2,dic)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_n2,Eq_bx,Eq_by,Eq_bz,Eq_b2x,Eq_b2y,Eq_b2z]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,C3,C4,C5,Symbol("psi"+str(order)),Symbol("psi"+str(order)+"_2b")))
        elif params.Bound_nb ==1:
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_bx,Eq_by,Eq_bz]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,Symbol("psi"+str(order))))

    elif condU == 'noslip':
        if params.Bound_nb ==1:
            U = (U.xreplace(makedic(veigen(eig,sol),order)))
            ut1 = U.dot(tx)
            ut2 = U.dot(ty)
            Eq_BU1= surfcond(ut1,dic)
            Eq_BU2 = surfcond(ut2,dic)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_BU1,Eq_BU2,Eq_bx,Eq_by,Eq_bz]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,C3,C4,Symbol("psi"+str(order))))
        elif params.Bound_nb ==2:
            un1 = U.dot(tx2)
            un2 = U.dot(ty2)
            Eq2_BU1= surfcond_2(un1,dic)
            Eq2_BU2 = surfcond_2(un2,dic)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_n2,Eq_BU1,Eq_BU2,Eq2_BU1,Eq2_BU2,Eq_bx,Eq_by,Eq_bz,Eq_b2x,Eq_b2y,Eq_b2z]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,Symbol("psi"+str(order)),Symbol("psi"+str(order)+"_2b")))

    elif condU == 'stressfree':
        if params.Bound_nb ==1:
            eu = strain(U)*nn
            eu1 = eu*tx
            eu2 = eu*ty
            Eq_BU1 = surfcond(eu1,dic,realtopo =False)
            Eq_BU2 = surfcond(eu2,dic,realtopo =False)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_BU1,Eq_BU2,Eq_bx,Eq_by,Eq_bz]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,C3,C4,Symbol("psi"+str(order))))
        elif params.Bound_nb ==2:
            eu = strain(U)*nn2
            eu1 = eu*tx2
            eu2 = eu*ty2
            Eq2_BU1 = surfcond2(eu1,dic,realtopo =False)
            Eq2_BU2 = surfcond2(eu2,dic,realtopo =False)
            TEq = [(taylor(eq,order,dic)).xreplace({x:0,y:0,t:0}) for eq in [Eq_n1,Eq_n2,Eq_BU1,Eq_BU2,Eq2_BU1,Eq2_BU2,Eq_bx,Eq_by,Eq_bz,Eq_b2x,Eq_b2y,Eq_b2z]]
            Mat, res = linear_eq_to_matrix(TEq,(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,Symbol("psi"+str(order)),Symbol("psi"+str(order)+"_2b")))

    Mat = Mat.evalf(mp.mp.dps)
    res = res.evalf(mp.mp.dps)
    Mat = mpmathM(Mat)
    res = mpmathM(res)
    try:
        abc = mp.qr_solve(Mat,res)[0]
    except:
        abc = mp.lu_solve(Mat,res)

    mantle =0 #In progress ...
    solans = zeros(7,1)
    for l in range(lso):
        solans = solans + abc[l]*Matrix(eig[l])*(ansatz).xreplace({kz:sol[l]})
    solans = solans.xreplace(dic)

    return(abc,solans,mantle)

# Create the variable with perturbative notation
def makeVar():
    if i > 1:
        lon = len(eigens)
        uu = zeros(3,1);bb = zeros(3,1)
    U = U0 *u0 + e**i*(Symbol('u'+str(i)+'x')*C.i + Symbol('u'+str(i)+'y')*C.j + Symbol('u'+str(i)+'z')*C.k)*ansatz0.xreplace({kxl:i*kxl,kyl:i*kyl,omega:i*omega})
    for h in range(1,i):
        for l in range(1,lon+1):
            uu = uu + e**h*(Matrix(eigens[l-1,:3,h-1]*ansatz0).xreplace({kz:Symbol("k" +str(h)+"_"+str(l))}).xreplace({kxl:h*kxl,kyl:h*kyl,omega:h*omega}))
        U = U + (uu[0]*C.i + uu[1]*C.j + uu[2]*C.k)+e**h*(Uparts[h])

    B = qAl*b0 +  e**i*(Symbol('b'+str(i)+'x')*C.i + Symbol('b'+str(i)+'y')*C.j + Symbol('b'+str(i)+'z')*C.k)*ansatz0.xreplace({kxl:i*kxl,kyl:i*kyl,omega:i*omega})
    for h in range(1,i):
        for l in range(1,lon+1):
            bb = bb + e**h*(Matrix(eigens[l-1,4:,h-1]*ansatz0).xreplace({kz:Symbol("k" +str(h)+"_"+str(l))}).xreplace({kxl:h*kxl,kyl:h*kyl,omega:h*omega}))
        B = B + (bb[0]*C.i + bb[1]*C.j + bb[2]*C.k)+e**h*Bparts[h]

    p = p0 +  e**i*Symbol('p'+str(i))*ansatz0.xreplace({kxl:i*kxl,kyl:i*kyl,omega:i*omega})
    for h in range(1,i):
        for l in range(1,lon+1):
            p = p + e**h*(eigens[l-1,3,h-1]*ansatz0.xreplace({kz:Symbol("k" +str(h)+"_"+str(l))}).xreplace({kxl:h*kxl,kyl:h*kyl,omega:h*omega}))
        p = p + +e**h*pparts[h]

    psi = psi0 + e**i*Symbol('psi'+str(i))*ansatz0.xreplace({kz:sqrt(-i*(kxl**2+kyl**2).xreplace(dico0),evaluate = False)})
    for h in range(1,i):
        psi = psi + e**h*psis[h]*ansatz0.xreplace({kz:sqrt(-h*(kxl**2+kyl**2).xreplace(dico0),evaluate = False)})

    if Bound_nb ==2:
        psi_2b = psi0_2b + (e**i*Symbol('psi'+str(i)+"_2b")*ansatz0.xreplace({kz:-sqrt(mp.mpf(str(-i*(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dico0))),evaluate =False)})).xreplace({kxl:i*kxl,kyl:i*kyl,omega:i*omega})
        for h in range(1,i):
            psi_2b = psi_2b + e**h*psis_2b[h]*ansatz0.xreplace({kz:-sqrt(mp.mpf(str(-h*(kxl**mp.mpf(2)+kyl**mp.mpf(2)).xreplace(dico0))),evaluate =False),kxl:h*kxl,kyl:h*kyl})

    if Bound_nb ==2:
        return(U,B,p,psi,psi_2b)
    if Bound_nb ==1:
        return(U,B,p,psi)

def subs_k(ex2,i,bnd):
    for o in range(0,i):
        for l in range(0,len(eigens)):
            ex2 = ex2.xreplace({Symbol("k" +str(o+1)+"_"+ str(l+1)):solk[l,o]})
    return(ex2)


#########################
###     Variables     ###
#########################

order = params.order
Bound_nb = params.Bound_nb

x = C.x
y = C.y
z = C.z

de = Del()
e,Dist = symbols("e,Dist",real = True)
Ri, Ro, Al, Rm, omega, t, kx,ky,kxl,kyl, kz,rho_r,alpha, g = symbols("Ri Ro Al Rm omega t kx,ky,kxl,kyl,kz,rho_r,alpha, g")
zeta,g0,g1,buoy,a,b,c,dv,ev,fv = symbols("zeta,g0,g1,buoy,a,b,c,dv,ev,gv")
BOx,BOy,BOz = symbols("BOx,BOy,BOz")
psi0 = symbols("psi0")
psi0_2b = symbols("psi0_2b")
for i in ['u','b']:
    for j in ['x','y','z']:
        globals() [i +'1'+j] = Symbol(i +'1'+j)
p0 = Symbol("p0")
b0x = Function("b0x")(x,y,z,t);b0y = Function("b0y")(x,y,z,t)
b0z = Function("b0z")(x,y,z,t);u0x = Function("u0x")(x,y,z,t)
u0y = Function("u0y")(x,y,z,t);u0z = Function("u0z")(x,y,z,t)
f0 = Function("f0")(x,y,z,t);f1 = Function("f1")(x,y,z,t)
f2 = Function("f2")(x,y,z,t);rho0 = Function("rho0")(x,y,z,t)
f0_2 = Function("f0_2")(x,y,z,t);f1_2 = Function("f1_2")(x,y,z,t)
f2_2 = Function("f2_2")(x,y,z,t)
g1 = Symbol("g1");Ax = Function("Ax")(x,y,z,t)
Ay = Function("Ay")(x,y,z,t);Az = Function("Az")(x,y,z,t)
Bx = Function("Bx")(x,y,z,t);By = Function("By")(x,y,z,t)
Bz = Function("Bz")(x,y,z,t)
AgradB =((Ax*D(Bx,x)+Ay*D(Bx,y)+Az*D(Bx,z))*C.i+
(Ax*D(By,x)+Ay*D(By,y)+Az*D(By,z))*C.j
+(Ax*D(Bz,x)+Ay*D(Bz,y)+Az*D(Bz,z))*C.k)

U0,qRm,qRe,qRo,qFr,chi,qAl,qRmm = symbols("U0,qRm,qRe,qRo,qFr,chi,qAl,qRmm",real = True)

u0 = u0x*C.i + u0y*C.j + u0z*C.k
b0 = b0x*C.i + b0y*C.j + b0z*C.k

ansatz0=exp(I*(omega*t+kxl*x+kyl*y+kz*z))
g = -g0*C.k
f = f0 + e*f1 #+ f2*e**2   Uncomment for smaller scales of topography

# Vector normal and tangential to the topography
delf = de(f)
nfo= delf/sqrt((delf&C.i)**2+(delf&C.j)**2+(delf&C.k)**2)
nx = (nfo&C.i)
ny = (nfo&C.j)
nz = (nfo&C.k)

tfox= (C.i)+((ny/nx)*C.j)+(-((nx**2+ny**2)/(nx*nz))*C.k)
tfox = tfox/sqrt((tfox&C.i)**2+(tfox&C.j)**2+(tfox&C.k)**2)
tfoy= (-(ny/nx)*C.i)+(C.j)+(0*C.k)
tfoy = tfoy/sqrt((tfoy&C.i)**2+(tfoy&C.j)**2+(tfoy&C.k)**2)
if Bound_nb == 2:
    f_2 = f0_2 + Dist + e*f1_2
    delf2 = de(f_2)
    nfo2= -delf2/sqrt((delf2&C.i)**2+(delf2&C.j)**2+(delf2&C.k)**2)
    tfox2= ((-(delf2&C.j)*C.i)+((delf2&C.i)*C.j)+(0*C.k))
    tfox = tfox/sqrt((tfox&C.i)**2+(tfox&C.j)**2+(tfox&C.k)**2)
    tfoy2= (((delf2&C.i)*C.i)+((delf2&C.j)*C.j)+(-(((delf2&C.i)**2+(delf2&C.j)**2)/(delf2&C.k))*C.j))
    tfoy = tfoy/sqrt((tfoy&C.i)**2+(tfoy&C.j)**2+(tfoy&C.k)**2)


######################
###     SCRIPT     ###
######################
#Parameters chosen for calculation
dico0 = params.dom
dico1 = params.dom1
# if dico1['qRe'] != 0 and condU =='Inviscid':
#     print("error Inviscid fluid incompatible with qRe != 0")
print(dico1)

lenvar = 1
Press = np.zeros((4,lenvar), dtype = np.complex)

#Choose what parameter you want to vary
for K,eta in enumerate(mp.linspace(-1,4,lenvar)):
    saveO = zeros(7,1) # save variable for speed, pressure and magnetic field (7 scalar)
    saveO_m = 0*C.i # magnetic field for the mantle part
    KXs = params.KXs
    KYs = params.KYs

    # Create the total topography and its normal vector
    topo_sum =0
    for kxx,kyy in zip(KXs,KYs):
        topo_sum += e*(zeta*(exp(I*(kxx*x+kyy*y))))/len(KXs)
        delfsum = de(z-topo_sum)
        nsum = delfsum/sqrt((delfsum&C.i)**2+(delfsum&C.j)**2+(delfsum&C.k)**2)
        nsum = nsum.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1})

    # Calculus for each Fourier component of the topography
    for KX,KY in zip(KXs,KYs):
        Usopart = 0
        solfull = 0
        so_part = zeros(7,1)
        psis = [0]
        psis_2b = [0]
        rhos = [0]
        Uparts = [0*C.i +0*C.j +0*C.k]
        Bparts = [0*C.i +0*C.j +0*C.k]
        pparts = [0]
        Usopart = 0*C.i +0*C.j +0*C.k
        Bsopart=0*C.i +0*C.j +0*C.k
        psopart = 0
        dico0[kyl] = KY # x wavenumber
        dico0[kxl] = KX # y wavenumber

        n = nfo.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1}) # normal vector
        tx = tfox.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1}) # 1st tangential vector
        ty = tfoy.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1}) # 2nd tangential vector

        # same for the 2nd boundary
        if Bound_nb == 2:
            n2 = nfo2.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1})
            tx2 = tfox2.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1})
            ty2 = tfoy2.xreplace({**dico0,**dico1}).doit().xreplace({**dico0,**dico1})

        # Calculus for each order i
        for i in range(1,order+1):
            print("ORDER",i)

            if Bound_nb ==2:
                U,B,p,psi,psi_2b = makeVar()
            if Bound_nb ==1:
                U,B,p,psi = makeVar()

            # Solve the mass conservation equation
            if i ==1:
                rho= rho0.xreplace({**dico0,**dico1})
            rho = rho + e**i*Symbol('rho'+str(i))*ansatz0.xreplace({kxl:i*kxl,kyl:i*kyl,omega:i*omega})
            print(rho)
            Eq_rho = diff(rho,t)+ U.dot(gradient(rho))
            print(Eq_rho)
            Eq_rho1 = taylor(Eq_rho,i,{**dico0,**dico1})
            print(Eq_rho1)
            r1 = list(solveset(Eq_rho1,Symbol('rho'+str(i))))[0]
            rho = rho.xreplace({Symbol('rho'+str(i)):r1})
            

            ansatz = (exp(I*(i*omega*t+i*kxl*x+i*kyl*y+kz*z))).xreplace({**dico0,**dico1})

            ### TEST OF ORDER0 ###
            # print("we test the order 0...")
            # M0,rme0,r10 = makeMatrix(U0 *u0,qAl*b0,p0,0,{**dico0,**dico1},vort=True)
            # print(rme0.xreplace({**dico0,**dico1}))

            M,rme,r1 = makeMatrix(U,B,p,i,{**dico0,**dico1},vort=True)
            M = M.xreplace({y:0})

            if i > 1:
                print("Particular solution")
                rmec = expand(rme)

                s= (expand(rmec[0])).args

                expo = []
                for st in s:
                    stt = str(st)
                    start = stt.find('exp')
                    expo = np.append(expo,stt[start:])
                expo = np.unique(expo)
                so_part = zeros(7,1)
                for ex in expo:
                    print(ex)
                    loc_dict = {'C':C}
                    ex = parse_expr(ex,local_dict = loc_dict)

                    coe = lambda x: x.coeff(ex)

                    rmep = (rmec).applyfunc(coe)
                    nwkz = simplify((log(ex).expand(force=True)/I).xreplace({x:0,y:0,t:0,z:1}))
                    Mp = simplify(M.xreplace({kz:nwkz}))

                    rmep = subs_k(rmep,i,Bound_nb)
                    Mp = subs_k(Mp,i,Bound_nb)
                    with mp.workdps(int(mp.mp.dps*2)):

                        Mp = mpmathM(Mp)
                        rmep = mpmathM(rmep)

                        try:
                            soluchap = mp.qr_solve(Mp,rmep)[0]
                        except:
                            soluchap = mp.lu_solve(Mp,rmep)
                            print('QR decomposition failed LU used')

                        sop = Matrix(soluchap)*ex
                        so_part = so_part+sop


                with mp.workdps(int(mp.mp.dps*2)):
                    so_part = so_part.xreplace({**dico0,**dico1})
                    Usopart = so_part[0]*C.i + so_part[1]*C.j + so_part[2]*C.k
                    Bsopart = so_part[4]*C.i + so_part[5]*C.j + so_part[6]*C.k
                    psopart = so_part[3]

                    Ubnd = U + e**i*Usopart
                    Bbnd = B + e**i*Bsopart
                    pbnd = p + e**i*psopart

                    Ubnd = subs_k(Ubnd,i,Bound_nb)
                    Bbnd = subs_k(Bbnd,i,Bound_nb)
                    pbnd = subs_k(pbnd,i,Bound_nb)

            ######  Homogeneous solution  #######

            print("Homogeneous solution")



            sol,eig,M1 = eigen(M,{**dico0,**dico1},i)
            #~ print(sol)
            if i == 1:
                Ubnd = U
                Bbnd = B
                eigens= np.zeros((len(sol),7,order),dtype = object)
                solk= np.zeros((len(sol),order),dtype = object)
            # print("res Eigens")
            # for s in range(len(eig)):
            #     print(max(np.array(M.xreplace({kz:sol[s]}).dot(eig[s]),dtype = np.complex)))
            abc,solhom,mantle = Bound(Ubnd,Bbnd,sol,eig,{**dico0,**dico1},i)

            for ei in range(len(sol)):
                eigens[ei,:,i-1] = abc[ei] * eig[ei]
            solk[:,i-1] = sol
            if Bound_nb == 1:
                psis.append(abc[-1])
            if Bound_nb ==2:
                psis.append(abc[-2])
                psis_2b.append(abc[-1])




            Uparts.append(Usopart)
            Bparts.append(Bsopart)
            pparts.append(psopart)
            solfull = so_part +solhom

            print("Finish")
            #######################
            ###   Final fields  ###
            #######################
            save = solfull
            save = subs_k(save,i,Bound_nb)
            saveO = saveO+(save*e**i).xreplace({**dico0,**dico1})

            # saveO_m= saveO_m+e*((abc[3]*C.i+abc[4]*C.j+abc[5]*C.k)*ansatz.xreplace({kz:-sqrt(-kxl**2-kyl**2-I*omega/qRmm)})).xreplace({**dico0,**dico1})
            #saveO_m= saveO_m+e*((abc[3]*C.i+abc[4]*C.j+abc[5]*C.k)*ansatz.xreplace({kz:I*1e12})).xreplace({**dico0,**dico1})
    ####################
    ###   PRESSURE   ###
    ####################
    # Eb = dico1[BOx]*C.i+dico1[BOy]*C.j+dico1[BOz]*C.k
    # Bf  = saveO[4]*C.i+saveO[5]*C.j+saveO[6]*C.k
    # PM = 1/2*((Eb*qAl+Bf).dot(Eb*qAl+Bf))
    #
    # avFp = (pressure(saveO[3],nsum,{**dico0,**dico1},1))
    # avPM = (pressure(PM,nsum,{**dico0,**dico1},1))
    # ft = mp.re(avFp-avPM)
    # print(ft)

    #######################
    ###   Dissipation   ###
    #######################
    # sm = 1000
    # sf =5e5
    # h_b = 100
    # V_b = mp.mpf('4e-5')
    # Om_b = mp.mpf('7.292e-5')
    # sig_b = (mp.mpf('1.00232')+1j*mp.mpf('2.5')*mp.mpf(1e-5))
    # L_b = 1e5
    # mu_b = 4*mp.pi*mp.mpf('1e-7')
    # rho_b = mp.mpf('1e4')
    # B_b = mp.mpf('5e-4')
    # N_b = mp.mpf('0.09')
    #
    #
    # b_f = ((saveO[4])*C.i+(saveO[5])*C.j+(saveO[6])*C.k)*mp.sqrt(rho_b*mu_b)*V_b
    # b_m = saveO_m*mp.sqrt(rho_b*mu_b)*V_b
    #
    # b_f = ((((b_f&C.i)+conjugate(b_f&C.i))*C.i + ((b_f&C.j)+conjugate(b_f&C.j))*C.j + ((b_f&C.k)+conjugate(b_f&C.k))*C.k)/2).xreplace({conjugate(x):x,conjugate(y):y,conjugate(z):z,conjugate(t):t})
    #
    # b_m = ((((b_m&C.i)+conjugate(b_m&C.i))*C.i + ((b_m&C.j)+conjugate(b_m&C.j))*C.j + ((b_m&C.k)+conjugate(b_m&C.k))*C.k)/2).xreplace({conjugate(x):x,conjugate(y):y,conjugate(z):z,conjugate(t):t})
    #
    # j_f = curl(b_f)/mu_b
    # j_m = curl(b_m)/mu_b
    #
    # Diss_f = 2*((j_f&C.i*conjugate(j_f&C.i))+(j_f&C.j*conjugate(j_f&C.j))+(j_f&C.k*conjugate(j_f&C.k)))/sf
    # Diss_m = 2*((j_m&C.i*conjugate(j_m&C.i))+(j_m&C.j*conjugate(j_m&C.j))+(j_m&C.k*conjugate(j_m&C.k)))/sm
    #
    # Fptay_f = (series(Diss_f,e,0,3).removeO()).xreplace({e:zeta,x:0,y:0,t:0}).xreplace(dico1)
    # Fptay_m  =(series(Diss_m,e,0,3).removeO()).xreplace({e:zeta,x:0,y:0,t:0}).xreplace(dico1)
    #
    #
    # FFp_f = lambdify(z,Fptay_f,'mpmath')
    # FFp_m = lambdify(z,Fptay_m,'mpmath')
    # print('integrate')
    #
    # avFp_f= mp.quad(FFp_f,[-mp.inf,0],maxdegree =12,verbose = True)
    # avFp_m= mp.quad(FFp_m,[0,mp.inf],maxdegree =12,verbose = True)
    #
    # print(avFp_f,avFp_m)
    # avFp = avFp_f+avFp_m
    # ft = mp.re(avFp)*1.52e14
    #
    # print(ft)

    # Press[0,K] = sol[0]
    # Press[1,K] = sol[1]
    # Press[2,K] = sol[2]
    # Press[3,K] = 10**(Omeg)


    #~ print(Press)
    # Tangential stress saving
    # np.savetxt('kz_Alfven_pi2.out',Press)

# Field Saving
# files= open("sol_topo_wave","w+")
# if Bound_nb ==2:
#     files.write(str({**dico1,**{"Bound":Bound_nb,'f1':str(((f-f0)).xreplace({**dico0,**dico1,**{C.x:Symbol('xx'),C.y:Symbol('yy')}})),'f2':str(((f_2-f0_2)).xreplace({**dico0,**dico1,**{C.x:Symbol('xx'),C.y:Symbol('yy')}}))}}))
# if Bound_nb ==1:
#     files.write(str({**dico1,**{"Bound":Bound_nb,'f1':str(((f-f0)).xreplace({**dico0,**dico1,**{C.x:Symbol('xx')}}))}}))
#
# files.write(str(saveO.xreplace({e:dico1[zeta]})))
# files.close()
