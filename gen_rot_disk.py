from EulerCommon import *
import pylab, os, argparse

parser = argparse.ArgumentParser(description='Rotation of the square.')
parser.add_argument("-t", type=float, default=.9, help="maximum time")
parser.add_argument("-N", type=int, default=100, help="number of points")
parser.add_argument("-D", type=int, default=30,
                    help="number of points on disk boundary")
args = parser.parse_args()

def gen_disk(k):
    t = np.linspace(0,2*np.pi,k+1);
    t = t[0:k]
    return np.vstack([np.cos(t),np.sin(t)]).T;


N = args.N # number of points
disk = gen_disk(args.D);
dens = ma.Density_2(disk);

X = ma.optimized_sampling_2(dens,N,niter=3);

pi = np.pi;
theta = np.pi*args.t
Y = np.zeros(X.shape)
Y[:,0] = np.cos(theta) * X[:,0] - np.sin(theta) * X[:,1]
Y[:,1] = np.sin(theta) * X[:,0] + np.cos(theta) * X[:,1]
euler_save("results/disk-T=%g-N=%d.npz" % (args.t,N), shape=disk, X=X, Y=Y);

