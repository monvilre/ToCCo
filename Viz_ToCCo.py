#! /usr/bin/env python3
# coding: utf-8
""" Tools for visualization with ToCCo output """
__author__ = "Rémy Monville"

import numpy as np
from sympy import *
import mpmath as mp
from mpmath import mpf
from sympy.vector import CoordSys3D,matrix_to_vector,curl,gradient,divergence,Del,Divergence,Gradient, laplacian,Curl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from cmcrameri import cm as SCM
import pickle


""" import for sympy """
U0,omega,qRe,chi,qRo,BOx,qAl,BOy,BOz,qRm,qRmm,qFr,Ri,zeta,Dist,e,t,p0,xx,yy,CUx,CUy,CUz,CBx,CBy,CBz,et,ev,tau0,qRmc,Rl = symbols("U0,omega,qRe,chi,qRo,BOx,qAl,BOy,BOz,qRm,qRmm,qFr,Ri,zeta,Dist,e,t,p0,xx,yy,CUx,CUy,CUz,CBx,CBy,CBz,et,ev,tau0,qRmc,Rl")
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
t = Symbol('t', real=True)
et,ev = symbols('et,ev')

def taylor(exp, nv,nt, dic):
    if nv==0 and nt==0:
        expt = exp.xreplace({ev: 0,et:0})
        return(expt)
    if nt == 0:
        df=diff((exp.xreplace(dic).doit()),ev,nv).xreplace({ev: 0,et:0})
    elif nv == 0:
        df=diff((exp.xreplace(dic).doit()),et,nt).xreplace({ev: 0,et:0})
    else:
        df = diff((exp.xreplace(dic).doit()),ev,nv,et,nt).xreplace({ev: 0,et:0})

    expt = (1/(factorial(nt)*factorial(nv)))*df
    return (expt)

def import_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    dic = data['meta']
    solfull = data['Expr']
    topo = data['topo']
    return(solfull,dic,topo)


def lambda_func(input):
    return(lambdify([x,y,z,t,ev,et],re(input)))
#
# def func_field(solfull,dic):
#     condB = dic['condB']
#     sux = solfull[0]
#     suy = solfull[1]
#     suz = solfull[2]
#     sp = solfull[3]
#     sbx = solfull[4]
#     sby =solfull[5]
#     sbz = solfull[6]
#     srho = solfull[7]
#     spsi  = solfull[8]
#
#     curlV = (curl(sux*C.i+suy*C.j))&C.k
#     sJ = curl(sbx*C.i+sby*C.j+sbz*C.k)
#
#     ffux = lambdify([x,y,z,t,ev,et],re(sux))
#     ffuy = lambdify([x,y,z,t,ev,et],re(suy))
#     ffuz = lambdify([x,y,z,t,ev,et],re(suz))
#     ffp =  lambdify([x,y,z,t,ev,et],re(sp))
#     ffbx = lambdify([x,y,z,t,ev,et],re(sbx))
#     ffby = lambdify([x,y,z,t,ev,et],re(sby))
#     ffbz = lambdify([x,y,z,t,ev,et],re(sbz))
#     ffrho = lambdify([x,y,z,t,ev,et],re(srho))
#     ffjx = lambdify([x,y,z,t,ev,et],re(sJ&C.i))
#     ffjy = lambdify([x,y,z,t,ev,et],re(sJ&C.j))
#     ffjz = lambdify([x,y,z,t,ev,et],re(sJ&C.k))
#
#     if condB == 'Thick':
#         sBm = spsi
#     elif condB == 'harm pot':
#         sBm = -gradient(spsi)
#     else:
#         print('Invalid magnetic BC')
#
#     ffpsix = lambdify([x,y,z,t,ev,et],re(sBm&C.i))
#     ffpsiy = lambdify([x,y,z,t,ev,et],re(sBm&C.j))
#     ffpsiz = lambdify([x,y,z,t,ev,et],re(sBm&C.k))
#
#     sJm = curl(sBm)
#     ffjmx = lambdify([x,y,z,t,ev,et],re(sJm&C.i))
#     ffjmy = lambdify([x,y,z,t,ev,et],re(sJm&C.j))
#     ffjmz = lambdify([x,y,z,t,ev,et],re(sJm&C.k))
#
#     # # taylorized V
#     # Tsux = taylor(solfull[0],1,1,{})*ev*et
#     # Tsuy = taylor(solfull[1],1,1,{})*ev*et
#     # Tsuz = taylor(solfull[2],1,1,{})*ev*et
#     # Tffux = lambdify([x,y,z,t,ev,et],re(Tsux))
#     # Tffuy = lambdify([x,y,z,t,ev,et],re(Tsuy))
#     # Tffuz = lambdify([x,y,z,t,ev,et],re(Tsuz))
#
#     TsP = taylor(solfull[3],0,1,{})*et +taylor(solfull[3],1,1,{})*et*ev
#     TffP = lambdify([x,y,z,t,ev,et],re(TsP))
#
#     f_field = [ffux,ffuy,ffuz,TffP,ffbx,ffby,ffbz,ffrho,ffpsix,ffpsiy,ffpsiz,ffjx,ffjy,ffjz,ffjmx,ffjmy,ffjmz]
#     return(f_field)

def Field(f_field,topo,lfunc_mantle,lfunc_core,X,Y,Z,T,Ev,Et):
    Var_tot = np.zeros(np.shape(X))
    mant = (Z-topo(X,Y,T,Et) >=0)
    cor = (Z-topo(X,Y,T,Et) <=0)
        Var_tot[mant] = lfunc_mantle(X[mant],Y[mant],Z[mant],T,Ev,Et)
        Var_tot[cor] = lfunc_core(X[cor],Y[cor],Z[cor],T,Ev,Et)
    return(Var_tot)


def cross_section_dim(f_field,topo,dic,X,Y,Z,time,zev,zeta,lfunc_mantle,lfunc_core,funcstream_mantle,funcstream_core,di = 'x',cmap = SCM.tokyo,**kwargs):
    "plot a cross section of choosen scalar field (var) + streamlines (varstream) "
    topo = lambdify([x,y,t,et],re(topo))
    rhoscale = np.float64(1e4) #scale considering rho = 1e4
    Xscale = np.float64(dic['Rl']*2890000)
    Vscale = np.float64(7.29e-5*Xscale*dic['Ro'])
    Bscale = np.float64(Vscale*np.sqrt(rhoscale*4*np.pi*1e-7)/dic['Al'])
    Jscale = np.float64(Bscale/(Xscale*4*np.pi*1e-7))
    Pscale = np.float64(Vscale**2*rhoscale)

    V = Field(f_field,topo,lfunc_mantle,lfunc_core,X,Y,Z,time,zev,zeta)
    # if varstream == 'U':
        # if di == 'x':
        #     streamU = Field(f_field,topo,funcstream_mantle,funcstream_core,X,Y,Z,time,zev,zeta)*Vscale
        # if di == 'y':
    streamU = Field(f_field,topo,funcstream_mantle[0],funcstream_core[0],X,Y,Z,time,zev,zeta)*Vscale
    streamV = Field(f_field,topo,funcstream_mantle[1],funcstream_core[1],X,Y,Z,time,zev,zeta)*Vscale
    # elif varstream == 'B':
    #     if di == 'x':
    #         streamU = Field(f_field,topo,'B_x',X,Y,Z,time,zev,zeta)*Bscale
    #     if di == 'y':
    #         streamU = Field(f_field,topo,'B_y',X,Y,Z,time,zev,zeta)*Bscale
    #     streamV = Field(f_field,topo,'B_z',X,Y,Z,time,zev,zeta)*Bscale
    # elif varstream == 'J':
    #     if di == 'x':
    #         streamU = Field(f_field,topo,'J_x',X,Y,Z,time,zev,zeta)*Jscale
    #     if di == 'y':
    #         streamU = Field(f_field,topo,'J_y',X,Y,Z,time,zev,zeta)*Jscale
    #     streamV = Field(f_field,topo,'J_z',X,Y,Z,time,zev,zeta)*Jscale

    # if var == 'U_x' or var == 'U_y' or var == 'U_z':
    #     V = V*Vscale
    #     Vscalestr = ' (m/s)'
    # elif var == 'B_x' or var == 'B_y' or var == 'B_z':
    #     V = V*Bscale*1e6
    #     Vscalestr = ' ($\mu$T)'
    # elif var == 'J_x' or var == 'J_y' or var == 'J_z':
    #     V = V*Jscale
    #     Vscalestr = ' (A/m$^{-2}$)'
    # elif var == 'rho':
    #     V = V*rhoscale
    #     Vscalestr = ' (kg/m$^3$)'
    # elif var == 'P':
    #     V = V*Pscale
    #     Vscalestr = ' (Pa)'
    # else:
    #     print('invalid variable for contourf plot')


    topography = topo(X[0],Y[0],time,zeta)*Xscale
    X = X*Xscale
    Y = Y*Xscale
    Z = Z*Xscale
    if di == 'x':
        MAIN = X
    if di == 'y':
        MAIN = Y
    MAIN=MAIN*1e-3

    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.nanmin(V)

    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.nanmax(V)


    plt.contourf(MAIN,Z,V,30,cmap = cmap,vmin = vmin,vmax=vmax)
    cb = plt.colorbar()
    cb.set_label('$'+var+'$' + Vscalestr, labelpad=10)
    #plt.contour(MAIN,Z,V,30,colors = 'k')
    if varstream != None:
        stream_X = MAIN[0]
        stream_Y = Z[:,0]
        # density = int(30*0.8)
        # start_X = np.linspace(np.min(stream_X),np.max(stream_X),density)
        # start_Y = np.linspace(np.min(stream_Y),np.max(stream_Y),density)
        # SX,SY =  np.meshgrid(start_X, start_Y)
        # start = np.array([SX.flatten(), SY.flatten()]).T

        # import streamplot as custom_stream
        # ax = plt.gca()
        # custom_stream.streamplot(ax,stream_X,stream_Y,streamU*1e-3,streamV,
        # color = "k",linewidth = 1,density = 0.793454,arrowsize = 0.8)

        plt.streamplot(stream_X,stream_Y,streamU*1e-3,streamV,color = "k",linewidth = 0.8,density = 1.4,arrowsize = 0.8)


    plt.plot(MAIN[0],topography,'k',linewidth = 3)

    plt.xlabel('$x$ (km)')
    plt.ylabel('$z$ (m)')
    plt.ylim(np.min(Z),np.max(Z))
    plt.xlim(np.min(MAIN),np.max(MAIN))


def plot_topo3D(topo,xn,yn,tn,zeta,dic,cmap = SCM.devon):
    topo = lambdify([x,y,t,et],re(topo))
    Xscale = np.float64(dic['Rl']*2890000)

    X,Y = np.meshgrid(xn,yn)


    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111, projection='3d')
    Z = topo(X,Y,tn,zeta)
    ax.contour(X*Xscale*1e-3, Y*Xscale*1e-3, Z*Xscale*1e-3, 10, colors = 'k', offset=-4*Xscale*1e-3*zeta)

    ax.set_zlim(-zeta*4*Xscale*1e-3,zeta*3.5*Xscale*1e-3)
    ax.view_init(30,)
    #ax.set_axis_off()
    ax.set_xlabel('$x$ (km)',labelpad =30)
    ax.set_ylabel('$y$ (km)',labelpad =30)
    ax.set_zlabel('$z$ (km)',labelpad =30)


    # Plot the surface.
    surf = ax.plot_surface(X*Xscale*1e-3, Y*Xscale*1e-3, Z*Xscale*1e-3, cmap=cmap,
                           linewidth=0, zorder=1,ccount = 70,rcount=70)
def plot_topo_cut(topo,xn,tn,zeta,dic,cmap = SCM.devon):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    topo = lambdify([x,y,t,et],re(topo))
    Xscale = np.float64(dic['Rl']*2890000)
    fig = plt.figure(figsize = (12,5))

    zn= np.real(topo(xn,0,tn,zeta))
    zn2 = np.real(topo(xn,np.pi,tn,zeta))

    xn = xn*Xscale*1e-3
    zn = zn*Xscale*1e-3
    zn2 = zn2*Xscale*1e-3

    plt.plot(xn,zn,'k',alpha=0.7)
    #plt.plot(xn,zn2,'k',alpha = 0.8)
    ym = zeta*Xscale*1.2*1e-3

    pa2 = np.array([xn,zn2]).T
    pa2 = np.vstack((np.array([np.min(xn),ym]),pa2))
    pa2 = np.vstack((pa2,np.array([np.max(xn),ym])))

    pa = np.array([xn,zn]).T
    pa = np.vstack((np.array([np.min(xn),ym]),pa))
    pa = np.vstack((pa,np.array([np.max(xn),ym])))

    path = Path(pa2)
    patch = PathPatch(path, facecolor='none',edgecolor='none')
    plt.gca().add_patch(patch)
    im = plt.imshow(xn.reshape(zn2.size,1),alpha = 0.8,  cmap=cmap,interpolation="bicubic",
                    origin='lower',extent=[-10*Xscale,10*Xscale,-1*Xscale*zeta*1e-3,ym],aspect="auto", clip_path=patch, clip_on=True)



    path = Path(pa)
    patch = PathPatch(path, facecolor='none',edgecolor='none')
    plt.gca().add_patch(patch)
    im = plt.imshow(xn.reshape(zn.size,1), cmap=cmap,interpolation="bicubic",
                    origin='lower',extent=[-10*Xscale,10*Xscale,-1*Xscale*zeta*1e-3,ym],aspect="auto", clip_path=patch, clip_on=True,alpha=1)



    plt.ylim(-zeta*Xscale*2*1e-3,ym)
    plt.xlim(np.min(xn),np.max(xn))
    plt.xlabel('$x$ (km)')
    plt.ylabel('$z$ (km)')
