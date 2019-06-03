#from seisflows.seistools import io
import numpy as np
#from seisflows.tools.array import grid2mesh,mesh2grid,gridsmooth,stack
from matplotlib import pyplot as plt
import copy
import scipy.signal as _signal
import scipy.interpolate as _interp
#from seisflows.postprocess.regularize import getmesh
def gauss2(X, Y, mu, sigma, normalize=True):
    """ Evaluates Gaussian over points of X,Y
    """
    # evaluates Gaussian over X,Y
    D = sigma[0, 0]*sigma[1, 1] - sigma[0, 1]*sigma[1, 0]
    B = np.linalg.inv(sigma)
    X = X - mu[0]
    Y = Y - mu[1]
    Z = B[0, 0]*X**2. + B[0, 1]*X*Y + B[1, 0]*X*Y + B[1, 1]*Y**2.
    Z = np.exp(-0.5*Z)

    if normalize:
        Z *= (2.*np.pi*np.sqrt(D))**(-1.)

    return Z

def stack(*args):
    return np.column_stack(args)

def gridsmooth(Z, span):
    """ Smooths values on 2D rectangular grid
    """
    import warnings
    warnings.filterwarnings('ignore')

    x = np.linspace(-2.*span, 2.*span, 2.*span + 1.)
    y = np.linspace(-2.*span, 2.*span, 2.*span + 1.)
    (X, Y) = np.meshgrid(x, y)
    mu = np.array([0., 0.])
    sigma = np.diag([span, span])**2.
    F = gauss2(X, Y, mu, sigma)
    F = F/np.sum(F)
    W = np.ones(Z.shape)
    Z = _signal.convolve2d(Z, F, 'same')
    W = _signal.convolve2d(W, F, 'same')
    Z = Z/W
    return Z


def mesh2grid(v, mesh):
    """ Interpolates from an unstructured coordinates (mesh) to a structured 
        coordinates (grid)
    """
    x = mesh[:,0]
    z = mesh[:,1]
    lx = x.max() - x.min()
    lz = z.max() - z.min()
    nn = v.size

    nx = np.around(np.sqrt(nn*lx/lz))
    nz = np.around(np.sqrt(nn*lz/lx))
    dx = lx/nx
    dz = lz/nz

    # construct structured grid
    x = np.linspace(x.min(), x.max(), nx)
    z = np.linspace(z.min(), z.max(), nz)
    X, Z = np.meshgrid(x, z)
    grid = stack(X.flatten(), Z.flatten())

    # interpolate to structured grid
    V = _interp.griddata(mesh, v, grid, 'linear')

    # workaround edge issues
    if np.any(np.isnan(V)):
        W = _interp.griddata(mesh, v, grid, 'nearest')
        for i in np.where(np.isnan(V)):
            V[i] = W[i]

    V = np.reshape(V, (int(nz), int(nx)))
    return V, grid
def grid2mesh(V, grid, mesh):
    """ Interpolates from structured coordinates (grid) to unstructured 
        coordinates (mesh)
    """
    return _interp.griddata(grid, V.flatten(), mesh, 'linear')

def read_fortran(filename):
    """ Reads Fortran style binary data and returns a numpy array.
    """
    with open(filename, 'rb') as file:                                                                              # read size of record
        file.seek(0)
        n = np.fromfile(file, dtype='int32', count=1)[0]

        # read contents of record
        file.seek(4)
        v = np.fromfile(file, dtype='float32')

    return v[:-1]


def write_fortran(v, filename):
    """ Writes Fortran style binary data. Data are written as single precision
        floating point numbers.
    """
    n = np.array([4*len(v)], dtype='int32')
    v = np.array(v, dtype='float32')

    with open(filename, 'wb') as file:
        n.tofile(file)
        v.tofile(file)
        n.tofile(file)


x=read_fortran('./proc000000_x.bin')
z=read_fortran('./proc000000_z.bin')
n=len(x)
vs=np.zeros((n,))
xm1=4200
amp=0.15
for i in range(n):
     if np.abs(x[i]-xm1)<500:
            vs[i]=4500-4500.0*amp*np.random.randn()
     else:
            vs[i]=5500
#vsrand = copy.copy(vs)
#for i in range(n):
#  vsrand[i]=vs[i]*amp*2.3*np.random.randn()
lamb = 4
mesh = stack(x, z)
vsrand,grid = mesh2grid(vs,mesh)
vsrand = gridsmooth(vsrand, lamb)
vsrand = grid2mesh(vsrand,grid,mesh)
#vs = vs+vsrand
#io.write_fortran(vs,'proc000000_vs.bin')
write_fortran(vsrand/2.0,'proc000000_rho.bin')
write_fortran(vsrand*1.7,'proc000000_vp.bin')
write_fortran(vsrand,'proc000000_vs.bin')
#write_fortran(x,'proc000000_x.bin')
#write_fortran(z,'proc000000_z.bin')
print x.min(),x.max(),z.min(),z.max()
#plt.figure(figsize=(10,6))
#sc=plt.tripcolor(x,z,vsrand,cmap='jet_r',vmin=3300,vmax=4000)
#
#plt.xlabel('x(km)')
#plt.ylabel('z(km)')
#plt.xlim([x.min(),x.max()])
#plt.ylim([z.min(),z.max()])
##sc.set_clim(3100,3900)
#plt.colorbar()
#plt.gca().invert_yaxis()
##plt.show()
#plt.savefig('invertvs.png')
