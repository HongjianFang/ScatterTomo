from seisflows.seistools import io
from seisflows.tools.array import mesh2grid,stack
import numpy as np
#from seisflows.postprocess.regularize import getmesh

x=io.read_fortran('../model_init/proc000000_x.bin')
z=io.read_fortran('../model_init/proc000000_z.bin')
mesh = stack(x, z)
#from seisflows.seistools.io import loadbin
vs = io.read_fortran('proc000000_vs.bin')
vs,grid = mesh2grid(vs,mesh)
print vs.shape
dd = vs[:,328]
np.savetxt('extrue.dat',dd,fmt='%7.4f')
#ext = vs[]
#io.write_fortran(x,'proc000000_x.bin')
#io.write_fortran(z,'proc000000_z.bin')

