import cudf

from cuml.neighbors import NearestNeighbors

from cuml.datasets import make_blobs

X, _ = make_blobs(n_samples=5, centers=5,

                  n_features=10, random_state=42)

# build a cudf Dataframe

X_cudf = cudf.DataFrame(X)

# fit model

model = NearestNeighbors(n_neighbors=3)

model.fit(X)
NearestNeighbors()

# get 3 nearest neighbors
print(X_cudf.shape)
distances, indices = model.kneighbors(X_cudf)

# print results

print(indices)


print(distances) 