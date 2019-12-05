# torch_neighbors
small python code to compute radius nearest neighbors on 3d point cloud using nanoflann and torch library.

## How to install
- You need to install pytorch (using pip or conda)
- execute
```
python setup.py build
```

Then in the build directory $PROJECT_PATH/build/lib.linux-x86_64-3.7 There is the .so file that you can import.
```
import neighbors
a = torch.randn(100000, 3) # only 3d is supported
res = neighbors.radius_search(a, a, 0.1, -1)
```
for more informations
```
help(neighbors.radius_search)
```



