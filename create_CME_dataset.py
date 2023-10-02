import pandas as pd
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.utils import shuffle
import shutil
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math

img_path = "Swansea/SWN001_SWN001_Maxillary_trios_export.stl"

# Create a new plot


# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file(img_path)

vec = your_mesh.vectors
print(vec[0,:,:])

print(your_mesh.vectors.shape)

figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()


figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()
"""
data = np.zeros(6, dtype=mesh.Mesh.dtype)

    # Top of the cube
data['vectors'][0] = np.array([[0, 1, 1],
                                  [1, 0, 1],
                                  [0, 0, 1]])
data['vectors'][1] = np.array([[1, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 1]])
    # Front face
data['vectors'][2] = np.array([[1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0]])
data['vectors'][3] = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 0]])
    # Left face
data['vectors'][4] = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 1]])
data['vectors'][5] = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [1, 0, 1]])

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
data['vectors'] -= .5

    # Generate 4 different meshes so we can rotate them later
meshes = [mesh.Mesh(data.copy()) for _ in range(4)]

    # Rotate 90 degrees over the Y axis
meshes[0].rotate([0.0, 0.5, 0.0], math.radians(90))

    # Translate 2 points over the X axis
meshes[1].x += 2

    # Rotate 90 degrees over the X axis
meshes[2].rotate([0.5, 0.0, 0.0], math.radians(90))
    # Translate 2 points over the X and Y points
meshes[2].x += 2
meshes[2].y += 2

    # Rotate 90 degrees over the X and Y axis
meshes[3].rotate([0.5, 0.0, 0.0], math.radians(90))
meshes[3].rotate([0.0, 0.5, 0.0], math.radians(90))
    # Translate 2 points over the Y axis
meshes[3].y += 2


    # Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d

    # Create a new plot
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

    # Render the cube faces
for m in meshes:
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

    # Auto scale to the mesh size
scale = np.concatenate([m.points for m in meshes]).flatten()
axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
pyplot.show()
"""