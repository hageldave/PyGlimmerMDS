# PyGlimmerMDS
Multidimensional scaling (MDS) for large data sets - a python implementation of the Glimmer algorithm.  
*[[Glimmer: Multilevel MDS on the GPU - 2009 - IEEE TVCG - Ingram, Munzner, Olano](https://doi.org/10.1109/TVCG.2008.85)]*

Glimmer performs dimensionality reduction on high-dimensional data sets of many instances, 
avoiding the quadratic runtime behavior of naive MDS implementations by employing a multilevel (coarse to fine) approach.
This implementation does **not** utilize the GPU, but gives considerable speedup nonetheless and makes MDS on large data
sets feasible.

Glimmer is a metric MDS and uses Euclidean distance in the high-dimensional space as the dissimilarity measure. 
This is **not** the classical MDS that has a linear projection solution.
Instead it solves the following optimization problem:

$$\underset{y_1,..,y_n}{\mathrm{argmin}} ~ \sum_{i=1}^n \sum_{j=i+1}^n \Big(\lVert x_i-x_j \rVert - \lVert y_i-y_j \rVert\Big) ^2 \quad \mathrm{where} x_i \in \mathbb{R}^D \mathrm{and} y_i \in \mathbb{R}^{d \ll D}$$


## Installation
PyGlimmerMDS is available on [PyPi](https://pypi.org/project/PyGlimmerMDS/) and can be installed through `pip`.
```
pip install PyGlimmerMDS
```
or if you want to install a specific commit use
```
pip install git+https://github.com/hageldave/PyGlimmerMDS@<commit_hash>
```

## How to use
### Very briefly
Performing Glimmer on a data set works like this:
```python
mds = Glimmer(decimation_factor=2, stress_ratio_tol=1-1e-5, rng=rng)
projection = mds.fit_transform(data) # alternative: projection, stress = execute_glimmer(data)
print(f"final stress={mds.stress}")
```

### Complete example
Jittering the Iris data set to produce a data set of 38,400 points. Performing Glimmer on this data set.
```python
from pyglimmermds import Glimmer, execute_glimmer
from sklearn import preprocessing as prep
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0xBA0BAB)

# get iris data
dataset = datasets.load_iris()
data = dataset.data
labels = dataset.target
# duplicate data with added noise
for _ in range(8):
  data = np.vstack((data,data+(rng.random((data.shape[0], data.shape[1]))*0.2-.1)))
  labels = np.append(labels,labels)
print(data.shape)
print(labels.shape)

# perform MDS
data = prep.StandardScaler().fit_transform(data)
mds = Glimmer(decimation_factor=2, stress_ratio_tol=1-1e-5, rng=rng)
projection = mds.fit_transform(data) # alternative: projection, stress = execute_glimmer(data)
print(f"final stress={mds.stress}")

# show scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.02)
ax.axis('equal')
plt.show(fig)
```
![glimmer_iris](https://github.com/user-attachments/assets/a1982cf9-59f9-4198-bb8a-984cdd41d210)
