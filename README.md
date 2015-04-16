# EulerSemidiscrete

This code goes with the article "Minimal geodesics along volume preserving maps through semi-discrete optimal transport", Q. Mérigot and J.M. Mirebeau.

## Installation

This code requires MongeAmpere and PyMongeAmpere, available here:

https://github.com/mrgt/MongeAmpere

https://github.com/mrgt/PyMongeAmpere

Before running any of the programs, you need to set PYTHONPATH to the right location:

``` sh
export PYTHONPATH=$PYTHONPATH:/path/to/PyMongeAmpere-build
```

## Running

On a simple example:
``` sh
mkdir results
python gen_rot_disk.py -t .5 -N 300
python euler_solve.py results/disk-T\=0.5-N\=300.npz 
python euler_view.py results/result-nt\=5-p\=0.666667-disk-T\=0.5-N\=300.npz
``` 

The parameters for more complex examples are described in the article.
