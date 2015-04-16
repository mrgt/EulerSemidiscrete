from EulerCommon import *
import random
import pylab
import os
import argparse

parser = argparse.ArgumentParser(description='View results of Euler equation.')
parser.add_argument('input', metavar='filename', type=str,
                   help='input file')
args = parser.parse_args()

fname = args.input
bname,ext = os.path.splitext(fname)

shape, S, Sproj = euler_load_result(fname)
S = Sproj
X = S[0]
N = X.shape[0]
bbox = bounding_box(X)
ii,jj,kk = cut_vertically(X)

def plot_timestep(X):
    sN = (5.0*10000.0/N);
    plt.cla()
    plt.scatter(X[ii,0], X[ii,1], s=sN, color='red');
    plt.scatter(X[jj,0], X[jj,1], s=sN, color='green');
    plt.scatter(X[kk,0], X[kk,1], s=sN, color='blue');
    plt.axis(bbox)
    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

I = random.sample(xrange(N), 30)
for i in I:
    x = S[:,i,0]
    plt.plot(S[:,i,0], S[:,i,1])
    plt.arrow(S[-1,i,0], S[-1,i,1], S[-1,i,0] - S[-2,i,0], S[-1,i,1] - S[-2,i,1])
plt.axis(bbox)
plt.axis("off")
pylab.savefig('%s-trajectories.jpg' % bname)
#plt.show()

os.system("rm /tmp/to-*.png")
for i in xrange(S.shape[0]):
    plot_timestep(S[i]);
    pylab.savefig('/tmp/to-%02d.png' % i, bbox_inches='tight', pad_inches = 0)
    plt.pause(.5)
os.system("convert -delay 60 -loop 0 /tmp/to-*.png %s.gif" % bname);
