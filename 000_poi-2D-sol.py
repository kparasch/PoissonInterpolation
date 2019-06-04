import numpy as np
import matplotlib.pyplot as plt
plt.style.use('kostas')
plt.close('all')

def charge_r(r):
    er = np.exp(r)
    if r > 0:
        return (1-er)/(1+er)/(2.*r) - er/(1.+er)**2
    else:
        return 0.5

def charge_true(x,y):
    shape = x.shape
    ret = np.empty_like(x)
    for ii in range(shape[0]):
        for jj in range(shape[1]):
            r = (x[ii,jj]**2 + y[ii,jj]**2)**0.5
            er = np.exp(r)
            if r > 0:
                ret[ii,jj] = (1-er)/(1+er)/(2.*r) - er/(1.+er)**2
            else:
                ret[ii,jj] = 0.5
    return ret

def efieldx_true(x,y):
    shape = x.shape
    ret = np.empty_like(x)
    if len(shape)==1:
        for ii in range(shape[0]):
            r = (x[ii]**2 + y[ii]**2)**0.5
            er = np.exp(r)
            if r > 0:
                ret[ii] = 0.5*x[ii]/r*(1.-er)/(1.+er)
            else:
                ret[ii] = 0.
        return ret
    for ii in range(shape[0]):
        for jj in range(shape[1]):
            r = (x[ii,jj]**2 + y[ii,jj]**2)**0.5
            er = np.exp(r)
            if r > 0:
                ret[ii,jj] = 0.5*x[ii,jj]/r*(1.-er)/(1.+er)
            else:
                ret[ii,jj] = 0.
    return ret

def phi_true(x,y):
    r = (x**2 + y**2)**0.5
    er = np.exp(r)
    return -r/2. + np.log(1+np.exp(r))

def fd(x,y,phi):

    dx = x[1,0]-x[0,0]
    dy = y[0,1]-y[0,0]
    dfdx=np.zeros_like(phi)
    dfdy=np.zeros_like(phi)
    dfdx[1:-1,:] = (-phi[:-2,:] + phi[2:,:])/(2*dx)
    dfdy[:,1:-1] = (-phi[:,:-2] + phi[:,2:])/(2*dy)
    return dfdx, dfdy

def poisson_sol(N,xmin=-20,xmax=20,yoff=0):

    ymin=xmin + yoff
    ymax=xmax + yoff

    x_sol = np.linspace(xmin,xmax,N)
    y_sol = np.linspace(ymin,ymax,N)
    h = x_sol[1]-x_sol[0]
    X_sol, Y_sol = np.meshgrid(x_sol,y_sol,indexing='ij')

    A = np.zeros([N**2,N**2],dtype=np.int)
    B = h**2 * charge_true(X_sol,Y_sol)
    B = B.reshape(N**2)
    for i in range(0,N):
        for j in range(0,N):
            xx = X_sol[i,j]
            yy = Y_sol[i,j]
            A[i+N*j,i+N*j] = 4
            if i == 0:
                B[i+N*j] += phi_true(xx - h, yy)
            else:
                A[i+N*j,i+N*j-1] = -1
            if i == N-1:
                B[i+N*j] += phi_true(xx + h, yy)
            else:
                A[i+N*j,i+N*j+1] = -1
            if j == 0:
                B[i+N*j] += phi_true(xx, yy - h)
            else:
                A[i+N*j,i+N*j-N] = -1
            if j == N-1:
                B[i+N*j] += phi_true(xx, yy + h)
            else:
                A[i+N*j,i+N*j+N] = -1

    Am1 = np.linalg.inv(A)
    phi_sol = np.dot(Am1, B)
    phi_sol = phi_sol.reshape(N,N)
    ex_sol, ey_sol = fd(X_sol, Y_sol, phi_sol)
    ex_sol=-ex_sol
    ey_sol=-ey_sol
    return X_sol, Y_sol, phi_sol, ex_sol, ey_sol

#def cubic_int(x,y,yp_exact=None,method='FD'):
#    
#    xmin = x[0]
#    xmax = x[len(x)-1]
#    h = x[1]-x[0]
#
#    if method=='FD':
#        yp=np.zeros_like(x)
#        yp[1:-1] = (-y[:-2] + y[2:])/(2)
#    elif method=='Exact':
#        yp = yp_exact*h
#
#    x_int = np.linspace(xmin,xmax,2000)
#    y_int = np.empty_like(x_int)
#    yp_int = np.empty_like(x_int)
#    for i,x in enumerate(x_int):
#        if x <= xmin+h or x >= xmax - h:
#            y_int[i] = 0
#            yp_int[i] = 0
#        else:
#            ix0 = int((x-xmin)/h)
#            ix1 = ix0+1
#            xt = (x - xmin)/h - ix0
#            a0 = y[ix0]
#            a1 = yp[ix0]
#            a2 = 3*(y[ix1]-y[ix0]) - (yp[ix1]+2*yp[ix0])
#            a3 = (yp[ix1] + yp[ix0]) -2*( y[ix1] - y[ix0] )
#            y_int[i] = a0 + a1*xt + a2*xt**2 + a3*xt**3
#            yp_int[i] =  (a1 + 2*a2*xt + 3*a3*xt**2)/h
#    
#    return x_int, y_int, -yp_int

yoff=0
x_true = np.linspace(-20,20,1000)
y_true = np.linspace(-20+yoff,20+yoff,1000)
h_true=x_true[1]-x_true[0]
X_true, Y_true = np.meshgrid(x_true,y_true, indexing='ij')
ch_true = charge_true(X_true, Y_true)
ex_true = efieldx_true(X_true, Y_true)
p_true = phi_true(X_true, Y_true)

#x_coarse = np.linspace(-20,20,20)
#ch_coarse = charge_true(x_coarse)
#e_coarse = efield_true(x_coarse)
#p_coarse = phi_true(x_coarse)
#e_fd = -fd(x_coarse, p_coarse)

N1=41
x_sol, y_sol, phi_sol, ex_sol, ey_sol = poisson_sol(N1,yoff=yoff)
h = x_sol[1,0]-x_sol[0,0]
#x_sol_fine, phi_sol_fine, e_sol_fine = poisson_sol_imp_glob(20,10)
#x_sol_g, phi_sol_g, e_sol_g  = poisson_sol_imp_glob(20,2)

#fig1 = plt.figure(1,figsize=(9,5))
#ax1 = fig1.add_subplot(1,1,1)
#cf1 = ax1.pcolormesh(X_true,Y_true, ch_true, cmap=plt.cm.RdBu, lw=0, rasterized=True)
#cf1.set_clim(0,-0.5)
#cbar1 = plt.colorbar(cf1, ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,0.])
#cbar1.set_label('$\\rho/\\epsilon$')


fig2 = plt.figure(2,figsize=(9,15))
ax21 = fig2.add_subplot(3,1,1)
cf21 = ax21.pcolormesh(X_true[1:,1:]-h_true/2., Y_true[1:,1:]-h_true/2., p_true[1:,1:], cmap=plt.cm.RdBu, lw=0, rasterized=True)
#cf21.set_clim(0,-0.5)
cbar21 = plt.colorbar(cf21)#, ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,0.])
cbar21.set_label('$\\phi$')
ax22 = fig2.add_subplot(3,1,2)
cf22 = ax22.pcolormesh(x_sol[1:,1:]-h/2., y_sol[1:,1:]-h/2., phi_sol[1:,1:]-phi_true(x_sol[1:,1:],y_sol[1:,1:]), cmap=plt.cm.RdBu, lw=0, rasterized=True)
#cf2.set_clim(0,-0.5)
cbar22 = plt.colorbar(cf22)#, ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,0.])
cbar22.set_label('$\\phi$')
ax23 = fig2.add_subplot(3,1,3)
ax23.plot(x_sol[:,N1//2], phi_sol[:,N1//2]-phi_true(x_sol[:,N1//2],y_sol[:,N1//2]),'bo-')
ax23.set_xlim(-5,5)
ax23.set_xlabel('$x$')
ax23.set_ylabel('$\\phi$')


fig3 = plt.figure(3,figsize=(9,15))
ax31 = fig3.add_subplot(3,1,1)
cf31 = ax31.pcolormesh(X_true[1:,1:]-h_true/2., Y_true[1:,1:]-h_true/2., ex_true[1:,1:], cmap=plt.cm.RdBu, lw=0, rasterized=True)
#cf21.set_clim(0,-0.5)
cbar31 = plt.colorbar(cf31)#, ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,0.])
cbar31.set_label('$E_x$')
ax32 = fig3.add_subplot(3,1,2)
cf32 = ax32.pcolormesh(x_sol[1:,1:]-h/2., y_sol[1:,1:]-h/2., ex_sol[1:,1:]-efieldx_true(x_sol[1:,1:],y_sol[1:,1:]), cmap=plt.cm.RdBu, lw=0, rasterized=True)
#cf2.set_clim(0,-0.5)
cbar32 = plt.colorbar(cf32)#, ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,0.])
cbar32.set_label('$E_x$')
ax33 = fig3.add_subplot(3,1,3)
ax33.plot(x_sol[:,N1//2], ex_sol[:,N1//2]-efieldx_true(x_sol[:,N1//2],y_sol[:,N1//2]),'bo-')
ax33.set_xlim(-5,5)
ax33.set_xlabel('$x$')
ax33.set_ylabel('$E_x$')
#ax3 = fig1.add_subplot(3,1,3)
#ax1.plot(x_true, ch_true)
#ax2.plot(x_true, e_true,'b',label='Exact Field')
#ax3.plot(x_true, p_true)
#ax2.plot(x_sol, e_sol, 'ko')
#ax2.plot(x_sol_fine, e_sol_fine, 'go')
#ax2.plot(x_sol_g, e_sol_g, 'ro')
#x_int_sol, phi_int_sol, e_int_sol = cubic_int(x_sol, phi_sol)
#x_int_sol_fine, phi_int_sol_fine, e_int_sol_fine = cubic_int(x_sol_fine, phi_sol_fine)
#x_int_sol_g, phi_int_sol_g, e_int_sol_g = cubic_int(x_sol_g, phi_sol_g)
#ax2.plot(x_int_sol, e_int_sol, 'k', label='Poisson Sol.')
#ax2.plot(x_int_sol_fine, e_int_sol_fine, 'g',label='Poisson Trick (x10p)' )
#ax2.plot(x_int_sol_g, e_int_sol_g, 'r',label='Poisson Trick (x2p)')
##ax3.plot(x_sol, phi_sol, 'ro')
#
#x_int, phi_int, e_int = cubic_int(x_coarse, p_coarse)
##ax2.plot(x_int, e_int, 'r',label='Interp with FD')
#ax3.plot(x_int, phi_int, 'r')
#
##ax2.plot(x_coarse, e_fd, 'ro')
#ax3.plot(x_coarse, p_coarse, 'ro')
#
#x_int_true, phi_int_true, e_int_true = cubic_int(x_coarse, p_coarse, -e_coarse,'Exact')
##ax2.plot(x_int_true, e_int_true, 'g',label='Interp. with Exact Derivs.')
##ax2.plot(x_coarse, e_coarse, 'go')
#ax3.plot(x_int_true, phi_int_true, 'g')
#
#
#ax2.set_xlim(-15,15)
#ax2.set_ylabel('E')
#ax2.set_xlabel('x')
#ax2.legend()
#
#fig4 = plt.figure(4,figsize=(15,15))
#ax4 = fig4.add_subplot(1,1,1)
#ax4.plot(x_true,e_true-e_true,'b',label='Exact Field')
##ax4.plot(x_int,e_int-efield_true(x_int),'r',label='Interp. with FD')
##ax4.plot(x_coarse,e_fd-efield_true(x_coarse),'ro')
##ax4.plot(x_int_true, e_int_true-efield_true(x_int_true), 'g',label='Interp. with Exact Derivs.')
##ax4.plot(x_coarse, e_coarse-efield_true(x_coarse), 'go')
#ax4.plot(x_int_sol,e_int_sol-efield_true(x_int_sol),'k',label='Poisson Sol.')
#ax4.plot(x_sol,e_sol-efield_true(x_sol),'ko')
#ax4.plot(x_int_sol_fine,e_int_sol_fine-efield_true(x_int_sol_fine),'m',label='Poisson Trick (x10p)')
#ax4.plot(x_sol_fine,e_sol_fine-efield_true(x_sol_fine),'mo')
#ax4.plot(x_int_sol_g,e_int_sol_g-efield_true(x_int_sol_g),'r',label='Poisson Trick (x2p)')
#ax4.plot(x_sol_g,e_sol_g-efield_true(x_sol_g),'ro')
#ax4.set_ylim(-6e-2,6e-2)
#ax4.set_xlim(-15,15)
#ax4.set_ylabel('E - E$_{exact}$')
#ax4.set_xlabel('x')
#ax4.legend()
plt.show()
