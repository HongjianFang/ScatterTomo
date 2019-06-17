#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : movedata.py
#
# Purpose :
#
# Creation Date : 01-06-2019
#
# Last Modified : Mon Jun 17 10:19:18 2019
#
# Created By : Hongjian Fang: hfang@mit.edu 
#
#_._._._._._._._._._._._._._._._._._._._._.*/
import os
nsrc = 30
datadir = 'wfs_25'
if not os.path.isdir(datadir):
  os.makedirs(datadir)
for ii in range(nsrc):
  srcdir = '000'+str(ii).zfill(2)
  os.system('cp ../scratch/solver/'+srcdir+'/traces/obs/Uy_file_single.su '+datadir+'/'+srcdir+'.su')
