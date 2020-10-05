import mpmath as mp
from sympy import exp,symbols,I,Symbol,Function,sqrt,conjugate
# u0x,u0y,u0z,t,b0x,b0y,b0z,rho0,alpha,x,y,z,kxl,kyl,f0,f1,f0_2,f1_2,g0,U0,omega,qRe,chi,qRo,BOx,BOy,BOz,qAl,qRm,qFr,Dist,zeta,Ri = symbols("""u0x,
# u0y,u0z,t,b0x,b0y,b0z,rho0,alpha,x,y,z,kxl,kyl,f0,f1,f0_2,f1_2,g0,U0,omega,qRe,chi,qRo,BOx,BOy,BOz,qAl,qRm,qFr,Dist,zeta,Ri""")
from sympy.vector import CoordSys3D
C = CoordSys3D('C')
x = C.x
y = C.y
z = C.z
t,alpha,kxl,kyl,g0,omega,BOx,BOy,BOz,zeta,Ri,p0 = symbols("""t,alpha,kxl,kyl,g0,omega,BOx,BOy,BOz,zeta,Ri,p0""")
U0,qRm,qRmm,qRe,qRo,qFr,chi,qAl,Dist = symbols("U0,qRm,qRmm,qRe,qRo,qFr,chi,qAl,Dist",real = True)
b0x = Function("b0x")(x,y,z,t);b0y = Function("b0y")(x,y,z,t)
b0z = Function("b0z")(x,y,z,t);u0x = Function("u0x")(x,y,z,t)
u0y = Function("u0y")(x,y,z,t);u0z = Function("u0z")(x,y,z,t)
f0 = Function("f0")(x,y,z,t);f1 = Function("f1")(x,y,z,t)
f2 = Function("f2")(x,y,z,t);rho0 = Function("rho0")(x,y,z,t)
f0_2 = Function("f0_2")(x,y,z,t);f1_2 = Function("f1_2")(x,y,z,t)
f2_2 = Function("f2_2")(x,y,z,t)

prec = 40
order =2
Bound_nb = 1

KXs = [1]#[1,1]
KYs = [0]#[1,-1]
##########################
###     Parameters     ###
##########################
#Imposed field and geometry
dom = {
	u0x:exp(I*omega*t),
	u0y:0,#I*exp(I*omega*t),
	u0z:0,
	b0x: BOx,
	b0y:0,
	b0z: BOz,
	rho0: (1+alpha*z),
	kxl :0,
	kyl : 0,
	f0: z,
	f1: -(zeta*exp(I*(kxl*x+kyl*y))),
	f0_2 :z,
	f1_2 : -zeta*(exp(I*(kxl*x+kyl*y))),
	Ri:-U0**2*qFr**2/(alpha*g0)#-U0**2*qFr**2/(alpha*g0)
	}
#Non dimensional parameters
#
dom1 = {
	U0:0,
	omega:1,
	qRe : 0,
	chi :mp.mpf('5.88e-3'),
	qRo :1/(mp.mpf('1.93e-4')),
	BOx :  0,
	BOy:0,
	qAl :1/(mp.mpf('2.24e-2')),
	BOz : 1,
	qRm :1/(mp.mpf('1.25')),
    qRmm:1/(mp.mpf('0.25')),
	qFr : 1e3,
	Dist :20,
	zeta:mp.mpf('0.001')}

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
# dom1= {
# 	U0: 0,
# 	omega:(Om_b*sig_b)*(L_b/V_b),
# 	qRe : 0,
# 	chi :0,
# 	qRo :1/(V_b/(L_b*Om_b)),
# 	BOx :0,
# 	BOy:0,
# 	qAl :1/((mp.sqrt(rho_b*mu_b)*V_b)/(B_b)),
# 	BOz :1,
# 	qRm :1/(V_b*L_b*sf*mu_b),
# 	qFr : 1/(V_b/(N_b*L_b)),
# 	qRmm :1/(V_b*L_b*sm*mu_b),
# 	Dist : 1,
#     zeta:(h_b/L_b)}



# dom1 = {
# 	U0:0,
# 	omega:1,
# 	qRe : 0,
# 	chi :mp.mpf('5.88e-3'),
# 	qRo :1/(mp.mpf('1.93e-4')),
# 	BOx :  0,
# 	BOy:0,
# 	qAl :1/(mp.mpf('2.24e-2')),
# 	BOz : 1,
# 	qRm :1/(mp.mpf('1.25')),
#     qRmm:1/(mp.mpf('0.25')),
# 	qFr : 1e3,
# 	Dist :20,
# 	zeta:mp.mpf('0.001')}
