import mpmath as mp
from sympy import exp, symbols, I, Symbol, Function, sqrt, conjugate
from sympy import Float
# u0x,u0y,u0z,t,b0x,b0y,b0z,rho0,alpha,x,y,z,kxl,kyl,f0,f1,f0_2,f1_2,g0,U0,omega,qRe,chi,qRo,BOx,BOy,BOz,qAl,qRm,qFr,Dist,zeta,Ri = symbols("""u0x,
# u0y,u0z,t,b0x,b0y,b0z,rho0,alpha,x,y,z,kxl,kyl,f0,f1,f0_2,f1_2,g0,U0,omega,qRe,chi,qRo,BOx,BOy,BOz,qAl,qRm,qFr,Dist,zeta,Ri""")
from sympy.vector import CoordSys3D
import sys
C = CoordSys3D('C')
x = C.x
y = C.y
z = C.z
t = Symbol('t', real=True)
alpha, kxl, kyl, g0, omega, BOx, BOy, BOz, zeta, Ri, p0 = symbols(
    """alpha,kxl,kyl,g0,omega,BOx,BOy,BOz,zeta,Ri,p0""")
U0, qRm, qRmm, qRmc, qRe, qRo, qFr, chi, qAl, Dist, ev, et, Rl = symbols(
    "U0,qRm,qRmm,qRmc,qRe,qRo,qFr,chi,qAl,Dist,ev,et,Rl", real=True)
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

prec = 80
order = 1
Bound_nb = 1
mp.mp.dps = prec

condB = "harm pot"  # "harm pot", "Thick"
condU = 'Inviscid'


buf = 0
atmo = 0
if atmo == 1:
    condB = "harm pot"

test = 0
######  Post process #####
pressure_stress = False
ohmic_dissipation = False

write_file = True
filename = "./sol_order1_Glane"

##########################
###     Parameters     ###
##########################

# lat_todo = [1e-2, 0.19634954, 0.39269908, 0.58904862, 0.78539816,
#        0.9817477 , 1.17809725, 1.37444679, 1.57079633]
# LAT =
# LAT = lat_todo[2]#mp.mpf('0.39269908') #In radian
QFR = 40  # (sys.argv[1])
LAT = mp.pi / 2
# LAT = 0.43973
# LAT = 4.39736842e-01
# Imposed field and geometry
dom = {
    # Zonal flow
    u0x:1,
    u0y:0,
    u0z:0,
    # Nutation forcing
    # u0x:ev*mp.sin(LAT)*(exp(I*omega*t)+exp(-I*omega*t))/2,
    # u0y:ev*(I*(exp(I*omega*t))-I*exp(-I*omega*t))/2,
    # u0z:0,
    # ### Test
    # u0x: ev * (exp(I * omega * t) + exp(-I * omega * t)) / 2,
    # u0y: 0,
    # u0z: 0,
    # Dipolar Field defined between -pi/2 pi/2
    # b0x :0,
    # b0y:mp.cos(LAT),
    # b0z: -2*mp.sin(LAT),
    b0x: 0,
    b0y: 0,
    b0z:1,  # linked to the expression of rho0 to solve equation of motion
    rho0: (1 - z),  # Warning alpha removed for devellopment
    f0: z,
    # 1D topo
    f1: -(exp(I * (x)) + exp(-I * (x))) / 2,
    # 1D topo respiration
    # f1 :-(exp(I*(x))+exp(-I*(x))+exp(I*(t))+exp(-I*(t))+exp(I*(omega*t))+exp(-I*(omega*t)))/2,
    # 1D topo 2wave
    # f1 :-(exp(I*(x))+exp(-I*(x))+1/6*exp(6*I*(x))+1/6*exp(-6*I*(x)))/(2+2*(1/6)),
    # f1 :-(exp(I*(x))+exp(-I*(x))+1/6*exp(6*I*(x))+1/6*exp(-6*I*(x))+exp(I*(y))+exp(-I*(y))+1/6*exp(6*I*(y))+1/6*exp(-6*I*(y)))/(4+4*(1/6)),
    # +exp(I*(omega*t))+exp(-I*(omega*t)))/2,
    # Egg box
    # f1: -(exp(I*(x+y)/(mp.sqrt(2)))+exp(-I*(x+y)/(mp.sqrt(2)))+exp(I*(x-y)/(mp.sqrt(2)))+exp(-I*(x-y)/(mp.sqrt(2))))/4,
    # f1: -(exp(I*(x+y))+exp(-I*(x+y))+exp(I*(x-y))+exp(-I*(x-y)))/4,
    # Egg box tilted
    # f1: -(exp(I*x)+exp(I*y)+exp(-I*x)+exp(-I*y))/4,
    f2: 0,  # -(exp(10*I*x)+exp(-10*I*x))/2,
    # f1:0,
    f0_2: z,
    f1_2: -(exp(2 * I * x) + exp(-2 * I * x)) / 2,
}

dom1 = {
    U0: 1,
    qRe: 0,

    qRo: 1 / (mp.mpf('1.93e-4')),
    chi: 30 / (1 / (mp.mpf('1.93e-4'))),
    BOx: 0,
    BOy: 0,
    qAl: 1 / (mp.mpf('2.24e-2')),
    BOz: 1,
    qRm: 1 / (mp.mpf('1.25')),
    qRmm: 1 / (mp.mpf('0.25')),
    qFr: mp.mpf('1e3'),
    Dist: 1}

Glane = {
    U0: 1,
    qRe: 0,
    omega: mp.mpf('182614.49999999997'),
    qRo: mp.mpf('2314.2857142857147'),
    Rl: mp.mpf('0.0054923930356456305'),
    # BOx : 0,
    qRmc: mp.nan,
    qAl: mp.mpf('11.627553482998906'),  # 1/(mp.mpf('2.24e-2')),
    # BOz :1,
    qRm: mp.mpf('0.10080000000000001'),  # 1/(mp.mpf('1.25')),
    qRmm: mp.mpf('10.0080000000000001'),  # 1/(mp.mpf('1.25')),
    qFr: mp.mpf('46285.71428571429'),  # mp.mpf('46285.71428571429'),
    # qFr : mp.mpf(str(QFR)),#mp.mpf('46285.71428571429'),
    Dist: 1}

Buffet2010 = {
    U0: 1,
    qRe: 0,
    omega: mp.mpf('182614.49999999994'),

    qRo: (mp.mpf('182249.99999999994')),
    Rl: mp.mpf('0.03460207612456747'),
    qAl: mp.mpf('111.50775725954817'),
    qRm: mp.mpf('0.39788735772973843'),
    qRmm: mp.mpf('198.9436788648692'),
    qRmc: mp.mpf('80'),
    qFr: mp.mpf('224999999.99999994'),
    Dist: 1e-2}

Me = {
    U0: 1,
    qRe: 0,
    omega: mp.mpf('182614.49999999994'),

    qRo: (mp.mpf('182249.99999999994')),
    Rl: mp.mpf('0.03460207612456747'),
    qAl: mp.mpf('111.50775725954817'),
    qRm: mp.mpf('0.39788735772973843'),
    qRmm: mp.mpf('198.9436788648692'),
    qRmc: mp.mpf('80'),
    # qFr : mp.mpf(QFR),
    Dist: 1e-2}


#     zeta:(h_b/L_b)}
