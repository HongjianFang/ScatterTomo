#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : movedata.py
#
# Purpose :
#
# Creation Date : 01-06-2019
#
# Last Modified : Sat Jun  1 18:10:30 2019
#
# Created By : Hongjian Fang: hfang@mit.edu 
#
#_._._._._._._._._._._._._._._._._._._._._.*/
import os
nsrc = 30
for ii in range(nsrc):
  srcdir = '000'+str(ii).zfill(2)
  os.makedirs(srcdir)
  os.system('cp ../scratch/solver/'+srcdir+'/traces/obs/Uy_file_single.su '+srcdir)
