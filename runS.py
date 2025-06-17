import numpy as np
from scipy.optimize import minimize
from numpy.linalg import svd, pinv, norm
import scipy.linalg as la

np.set_printoptions(linewidth=160,precision=5,suppress=True)

nx = 4;
xs = np.array([-0.9, -0.2, 0.2, 0.9])
ws = np.ones_like(xs);

STD = 1e-4;

beta = 100;
nz = 256;
hf = nz//2;
hs = np.pi/beta * np.arange(1,2*hf,2) * 1j
zs = np.concatenate([hs,-hs]);

def gfn(t,s):
    return np.reciprocal(t[:,None]-s[None,:])

EPS = STD
ng = 1024
gs = np.sort( np.cos(np.pi * np.arange(0,ng+1)/ng) )
T = gfn(zs,gs);

#normalize the columns
column_norms = norm(T, axis=0)
T = T / column_norms
M = T @ np.diag(gs) @ pinv(T, rcond=EPS)
tmp = norm(M@T-T@np.diag(gs),2)/norm(M@T,2)
print("MT-TD error", "%.2E"%tmp);

us = gfn(zs,xs) @ ws
us *= 1 + STD*(np.random.randn(nz) + 1j*np.random.randn(nz))

na = nx+1;
A = np.zeros((nz,na), dtype=complex)
A[:,0] = us;
for  g in range(1,na):
    A[:,g] = M @ A[:,g-1]

tU,tS,tV = svd(A, full_matrices=False)
p = tV[-1,:]
rts = np.roots(p[::-1])

rts = np.real(rts)
bad = np.where(np.abs(rts)>1)
rts[bad] = rts[bad] / np.abs(rts[bad])

xa = np.sort(rts);
tmp = gfn(zs,xa);
tmpA = np.vstack((np.real(tmp),np.imag(tmp)))
tmpb = np.concatenate((np.real(us),np.imag(us)))
wa = np.linalg.lstsq(tmpA,tmpb)[0]

def fun(y):
    pos = y[0:nx]
    wgt = y[nx:]
    return norm(gfn(zs,pos)@wgt - us)**2

result = minimize(fun, np.concatenate((xa,wa)))
tmp = result.x
xb = tmp[0:nx]
wb = tmp[nx:]

vb = gfn(zs,xb) @ wb
va = gfn(zs,xa) @ wa
vs = gfn(zs,xs) @ ws

relerr = norm(vb-vs)/norm(vs)
err1=norm(va-us)/norm(us)
err2=norm(vb-us)/norm(us)
print("relerr: ", "%.2E"%err1, "%.2E"%err2) 

    



    
