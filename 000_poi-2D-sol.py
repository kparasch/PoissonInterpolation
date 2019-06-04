import numpy as np
import matplotlib.pyplot as plt
plt.style.use('kostas')
plt.close('all')

def charge_true(x,y):
    r = (x**2 + y**2)**0.5
    er = np.exp(r)
    return (1-er)/(1+er)/(2.*r) - er/(1.+er)**2

def ex_true(x,y):
    r = (x**2 + y**2)**0.5
    er = np.exp(r)
    return 0.5*x/r*(1.-er)/(1.+er)

def phi_true(x,y):
    r = (x**2 + y**2)**0.5
    er = np.exp(r)
    return -r/2. + np.log(1+np.exp(r))

def fd(x,y,phi):

    dx = x[1,0]-x[0,0]
    dy = y[0,1]-y[0,0]
    ex=np.zeros_like(x)
    ex[1:-1,:] = (-phi[:-2,:] + phi[2:,:])/(2*dx)
    ey[:,1:-1] = (-phi[:,:-2] + phi[:,2:])/(2*dy)
    return ex, ey

def cubic_int(x,y,yp_exact=None,method='FD'):
    
    xmin = x[0]
    xmax = x[len(x)-1]
    h = x[1]-x[0]

    if method=='FD':
        yp=np.zeros_like(x)
        yp[1:-1] = (-y[:-2] + y[2:])/(2)
    elif method=='Exact':
        yp = yp_exact*h

    x_int = np.linspace(xmin,xmax,2000)
    y_int = np.empty_like(x_int)
    yp_int = np.empty_like(x_int)
    for i,x in enumerate(x_int):
        if x <= xmin+h or x >= xmax - h:
            y_int[i] = 0
            yp_int[i] = 0
        else:
            ix0 = int((x-xmin)/h)
            ix1 = ix0+1
            xt = (x - xmin)/h - ix0
            a0 = y[ix0]
            a1 = yp[ix0]
            a2 = 3*(y[ix1]-y[ix0]) - (yp[ix1]+2*yp[ix0])
            a3 = (yp[ix1] + yp[ix0]) -2*( y[ix1] - y[ix0] )
            y_int[i] = a0 + a1*xt + a2*xt**2 + a3*xt**3
            yp_int[i] =  (a1 + 2*a2*xt + 3*a3*xt**2)/h
    
    return x_int, y_int, -yp_int

def poisson_sol_imp_glob(N,kk,xmin=-20,xmax=20):

    x_sol = np.linspace(xmin,xmax,N)
    x_sol_imp = np.linspace(xmin,xmax,kk*(N-1)+1)
    h = x_sol[1]-x_sol[0]
    h_imp = x_sol_imp[1]-x_sol_imp[0]
    B = - h_imp**2 * charge_true(x_sol)
    B_imp = np.empty_like(x_sol_imp)
    #B[0] -= phi_true(xmin-h)
    #B[N-1] -= phi_true(xmax+h)
    for i in range(N-1):
        B_imp[kk*i]=B[i]
        for j in range(1,kk):
            B_imp[kk*i+j] = (B[i+1]-B[i])*j/kk + B[i]
    B_imp[kk*(N-1)]=B[N-1]
    B_imp[0] -= phi_true(xmin-h_imp)
    B_imp[kk*(N-1)] -= phi_true(xmax+h_imp)
    A = np.eye(kk*(N-1)+1,k=1) -2*np.eye(kk*(N-1)+1) + np.eye(kk*(N-1)+1,k=-1)
    Am1 = np.linalg.inv(A)
    phi_sol = np.dot(Am1, B_imp)
    e_sol=np.zeros_like(phi_sol)
    e_sol[1:-1] = (phi_sol[:-2] - phi_sol[2:])/(2*h_imp)
    return x_sol_imp, phi_sol, e_sol

def poisson_sol(N,xmin=-20,xmax=20):

    x_sol = np.linspace(xmin,xmax,N)
    h = x_sol[1]-x_sol[0]
    B = - h**2 * charge_true(x_sol)
    B[0] -= phi_true(xmin-h)
    B[N-1] -= phi_true(xmax+h)
    A = np.eye(N,k=1) -2*np.eye(N) + np.eye(N,k=-1)
    Am1 = np.linalg.inv(A)
    phi_sol = np.dot(Am1, B)
    e_sol=np.zeros_like(phi_sol)
    e_sol[1:-1] = (phi_sol[:-2] - phi_sol[2:])/(2*h)
    return x_sol, phi_sol, e_sol

x_true = np.linspace(-20,20,1000)
ch_true = charge_true(x_true)
e_true = efield_true(x_true)
p_true = phi_true(x_true)

x_coarse = np.linspace(-20,20,20)
ch_coarse = charge_true(x_coarse)
e_coarse = efield_true(x_coarse)
p_coarse = phi_true(x_coarse)
e_fd = -fd(x_coarse, p_coarse)


fig1 = plt.figure(1,figsize=(15,15))
ax1 = fig1.add_subplot(3,1,1)
fig2 = plt.figure(2,figsize=(15,15))
ax2 = fig2.add_subplot(1,1,1)
ax3 = fig1.add_subplot(3,1,3)
ax1.plot(x_true, ch_true)
ax2.plot(x_true, e_true,'b',label='Exact Field')
ax3.plot(x_true, p_true)

x_sol, phi_sol, e_sol = poisson_sol(20)
x_sol_fine, phi_sol_fine, e_sol_fine = poisson_sol_imp_glob(20,10)
x_sol_g, phi_sol_g, e_sol_g  = poisson_sol_imp_glob(20,2)
ax2.plot(x_sol, e_sol, 'ko')
ax2.plot(x_sol_fine, e_sol_fine, 'go')
ax2.plot(x_sol_g, e_sol_g, 'ro')
x_int_sol, phi_int_sol, e_int_sol = cubic_int(x_sol, phi_sol)
x_int_sol_fine, phi_int_sol_fine, e_int_sol_fine = cubic_int(x_sol_fine, phi_sol_fine)
x_int_sol_g, phi_int_sol_g, e_int_sol_g = cubic_int(x_sol_g, phi_sol_g)
ax2.plot(x_int_sol, e_int_sol, 'k', label='Poisson Sol.')
ax2.plot(x_int_sol_fine, e_int_sol_fine, 'g',label='Poisson Trick (x10p)' )
ax2.plot(x_int_sol_g, e_int_sol_g, 'r',label='Poisson Trick (x2p)')
#ax3.plot(x_sol, phi_sol, 'ro')

x_int, phi_int, e_int = cubic_int(x_coarse, p_coarse)
#ax2.plot(x_int, e_int, 'r',label='Interp with FD')
ax3.plot(x_int, phi_int, 'r')

#ax2.plot(x_coarse, e_fd, 'ro')
ax3.plot(x_coarse, p_coarse, 'ro')

x_int_true, phi_int_true, e_int_true = cubic_int(x_coarse, p_coarse, -e_coarse,'Exact')
#ax2.plot(x_int_true, e_int_true, 'g',label='Interp. with Exact Derivs.')
#ax2.plot(x_coarse, e_coarse, 'go')
ax3.plot(x_int_true, phi_int_true, 'g')


ax2.set_xlim(-15,15)
ax2.set_ylabel('E')
ax2.set_xlabel('x')
ax2.legend()

fig4 = plt.figure(4,figsize=(15,15))
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(x_true,e_true-e_true,'b',label='Exact Field')
#ax4.plot(x_int,e_int-efield_true(x_int),'r',label='Interp. with FD')
#ax4.plot(x_coarse,e_fd-efield_true(x_coarse),'ro')
#ax4.plot(x_int_true, e_int_true-efield_true(x_int_true), 'g',label='Interp. with Exact Derivs.')
#ax4.plot(x_coarse, e_coarse-efield_true(x_coarse), 'go')
ax4.plot(x_int_sol,e_int_sol-efield_true(x_int_sol),'k',label='Poisson Sol.')
ax4.plot(x_sol,e_sol-efield_true(x_sol),'ko')
ax4.plot(x_int_sol_fine,e_int_sol_fine-efield_true(x_int_sol_fine),'m',label='Poisson Trick (x10p)')
ax4.plot(x_sol_fine,e_sol_fine-efield_true(x_sol_fine),'mo')
ax4.plot(x_int_sol_g,e_int_sol_g-efield_true(x_int_sol_g),'r',label='Poisson Trick (x2p)')
ax4.plot(x_sol_g,e_sol_g-efield_true(x_sol_g),'ro')
ax4.set_ylim(-6e-2,6e-2)
ax4.set_xlim(-15,15)
ax4.set_ylabel('E - E$_{exact}$')
ax4.set_xlabel('x')
ax4.legend()
plt.show(False)
