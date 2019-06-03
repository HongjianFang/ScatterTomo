#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

src = pd.read_table('/Users/hongjianfang/study/ScatterImaging/checkers/specfem2d-scatter/DATA/sources.dat',delim_whitespace=True,names=['x','z'])
rc = pd.read_table('/Users/hongjianfang/study/ScatterImaging/checkers/specfem2d-scatter/DATA/STATIONS',delim_whitespace=True,names=['sname','net','x','z','cc1','cc2'])

f=open('proc000000_x.bin',"rb")
x=np.fromfile(f,dtype=np.float32)
f=open('proc000000_z.bin',"rb")
z=np.fromfile(f,dtype=np.float32)
f=open('proc000000_vs.bin',"rb")
vs=np.fromfile(f,dtype=np.float32)

plt.figure(figsize=(10,6))
sc=plt.tripcolor(x,z,vs,cmap='jet_r')#,vmin=4500,vmax=5500)

plt.plot(src['x'],src['z'],'r*')
plt.plot(rc['x'],rc['z'],'yv')
plt.xlabel('x(km)')
plt.ylabel('z(km)')
plt.xlim([x.min(),x.max()])
plt.ylim([z.min(),z.max()])
#sc.set_clim(3100,3900)
plt.colorbar()
#plt.gca().invert_yaxis()
#plt.show()
plt.savefig('invertvs.png')
