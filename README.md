# torch_neighbors
small python code to compute radius nearest neighbors on 3d point cloud using nanoflann and  c++ Pytorch library.

## How to install
- You need to install pytorch (using pip or conda)
- execute (It is better to create a virtual environment before executing this command):

```
python setup.py install
```
then on a python script
```python
import neighbors
a = torch.randn(100000, 3) # only 3d is supported
ind_neigh, dist_neigh = neighbors.radius_search(a, a, 0.1, -1)
```
for more information
```python
help(neighbors.radius_search)
```
