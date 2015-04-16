from EulerCommon import *
import pylab
import os
import argparse

parser = argparse.ArgumentParser(description='Solve 2D Euler equation.')
parser.add_argument('input', metavar='filename', type=str,
                   help='input file')
parser.add_argument("-p", type=float, default=2./3.,
                    help="exponent for penalization")
parser.add_argument("-k", type=int, default=2,
                    help="log_2(number of time steps)")
parser.add_argument("-sigma", type=float, default=0,
                    help="perturbation of initial midpoint, for singular situations")
args = parser.parse_args()
p = args.p
k = args.k

fname = args.input
shape, X, Y = euler_load_experiment(fname)
hname,tname = os.path.split(fname)
t = np.power(2,k) + 1
oname = os.path.join(hname, "result-nt=%d-p=%g-%s" % (t, p, tname)) 
print oname

dens = ma.Density_2(shape);
S = euler_solve_lbfgs(shape,X,Y,p=p,k=k,sigma=args.sigma)

# project S onto incompressible
t = S.shape[0]
N = S.shape[1]
Sproj = np.zeros((t,N,2))
for i in xrange(t):
    Sproj[i] = project_on_incompressible(dens, S[i])

euler_save(oname, shape=shape, S=S, Sproj=Sproj);
