# PyGlimmerMDS
A python implementation of the [Glimmer algorithm](https://doi.org/10.1109/TVCG.2008.85) for multidimensional scaling (MDS).

Glimmer performs dimensionality reduction on high-dimensional data sets of many instances, 
avoiding the quadratic runtime behavior of naive MDS implementations by employing a multilevel (coarse to fine) approach.
This implementation does **not** utilize the GPU, but gives considerable speedup nonetheless and makes MDS on large data
sets feasible.

Glimmer is a metric MDS and uses Euclidean distance in the high-dimensional space as the dissimilarity measure.


## Installation
```
pip install PyGlimmerMDS
```
or if you want to install a specific commit use
```
pip install git+https://github.com/hageldave/PyGlimmerMDS@<commit_hash>
```

## How to use
Jittering the Iris data set to produce a data set of 38,400 points. Performing Glimmer on this data set.

```python
from pyglimmermds import Glimmer, execute_glimmer
from sklearn import preprocessing as prep
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# get iris data
dataset = datasets.load_iris()
data = dataset.data
labels = dataset.target
# duplicate data with added noise
for _ in range(8):
  data = np.vstack((data,data+(np.random.rand(data.shape[0], data.shape[1])*0.2-.1)))
  labels = np.append(labels,labels)
print(data.shape)
print(labels.shape)
# perform MDS
data = prep.StandardScaler().fit_transform(data)
mds = Glimmer(decimation_factor=2, stress_ratio_tol=1 - 1e-5)
projection = mds.fit_transform(data) # alternative: execute_glimmer(data)
# show scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=1)
ax.axis('equal')
plt.show(fig)
```
![glimmer_iris](https://github.com/user-attachments/assets/8dad7f6b-0f08-4088-b76f-edd572a7f886)
