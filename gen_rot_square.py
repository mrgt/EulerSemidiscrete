# recover the "rotation" of a square described in "Reconstruction
# d'ecoulements incompressibles a partir de donnees lagrangiennes",
# Y. Brenier & Michel Roesch.

from EulerCommon import *
import pylab, os, argparse


parser = argparse.ArgumentParser(description='Rotation of the square.')
parser.add_argument("-t", type=float, default=.9,
                    help="maximum time")
parser.add_argument("-N", type=int, default=100,
                    help="number of points")
args = parser.parse_args()

N = args.N # number of points
t = args.t # maximum time

square=np.array([[0.,0.],[0.,1.],[1.,1.],[1.,0.]]);
dens = ma.Density_2(square);

X = ma.optimized_sampling_2(dens,N,niter=3);

pi = np.pi;
nt = 10000
dt = t/nt
Y = X.copy()
T = np.linspace(1,nt-1,30).astype(int).tolist();
T.append(nt-1)

for i in xrange(1,nt):
    print "i=%d\r" % i,
    if i in T:
        Y = project_on_incompressible(dens,Y)
    u = np.cos(pi*Y[:,0]) * np.sin(pi*Y[:,1]);
    v = np.cos(pi*Y[:,1]) * np.sin(pi*Y[:,0]);
    Y[:,0] = Y[:,0] + dt * v;
    Y[:,1] = Y[:,1] - dt * u;

euler_save("results/square-T=%g-N=%d.npz" % (t,N), shape=square, X=X, Y=Y);

