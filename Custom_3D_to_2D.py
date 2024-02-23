"""
 implementation of transforming 3D surface (point cloud) to 2D pixel intensity values.
* Date: 22 December, 2021
* Author: Dr. Ekpo Otu


## ALGOTRITHM
Below is the step-by-step process to achieve the above task!

- **STEP 1:** Load a 3D point cloud, i.e.(x,y,z) coordinates.

- **STEP 2:** Scale (i.e. ***normalize***) the point cloud data in STEP 1, and ***center*** it on its centroid (mean).

- **STEP 3:** Get the Minimum and Maximum of each point cloud coordinate.

- **STEP 4:** Create a 2D grid of say, $NxN$ or $NxM$ with $(x,y)$ plane. Where $N=200$, $250$, $400$, or $500$, etc. Where the values in x-axis contains linearly-spaced floating point values between minX and maxX of x-coordinate in the 3D xyz point cloud, and the values in y-axis contains linearly-spaced floating point values between minY and maxY of y-coordinate in the 3D xyz point cloud.

- **STEP 5:** Get all the (x,y)-axes values of the 2D meshgrid created in STEP 4, derived with np.linspace(minX, maxX, N) and np.linspace(minY, maxY, N).

- **STEP 6:** Obtain the ***centroid*** of the 2D meshgrid's 'xyValues', to enable us work out th threshold value.

- **STEP 7:** Translate (or center) the 3D point cloud data (note, we're only interested in its xy-coords) to the center of the 2D xy-axes or xvValues.

- **STEP 8:** Compute intensity values for each 2D grid, i.e. for ***xyValues***, using NN distance of 3 clossest point on the ***pointcloud_translated*** data.
- **STEP 9:** Final step to convert INTENSITY value into 2D grey scale image. TBC...
"""

# Import needed libraries and utility functions
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
from scipy.spatial import cKDTree
import trimesh
from PIL import Image
import os
#import gc
from numba import jit, cuda
import cudf, cuml
import time
import warnings
warnings.filterwarnings('ignore')
# np.set_printoptions(suppress = True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#cp.cuda.Device(1).use()
print(os.environ["CUDA_VISIBLE_DEVICES"])
#cuml.cuda.Device(1).use()
'''
# PHASE 2
NOW BUILD ALL OF THE ABOVE TO RUN FOR A NUMBER OF MODELS IN A FOLDER
'''

#MODIFIED by Ekpo on Monday, 11th February, 2019.
#Function to Normalize [numpy 1D array].
#@jit(target_backend='cuda')
def normalize1Darrayx(data):
    '''
    NOTE: Adding together the values returned will NOT SUM to 1.
    data - Must be an [N x 1] or [1 x N] array.

    Author: Ekpo (eko@aber.ac.uk)  25th February, 2019
    '''

    return (data - cp.min(data)) / (cp.max(data) - cp.min(data))

def normalize1Darrayx2(data):
    data = cp.array(data)
    zero_array = cp.zeros((data.shape))
    for row in range(0, data.shape[0]):
        zero_array[row] = (data[row] - np.min(data)) / (np.max(data) - np.min(data))
    
    return zero_array



    
#Function that uses scipy library's 'FASTER' cKDTree to find k-Nearest neighbours to point p of interest
def k_nearestNeigbourV2(points3D_xy, points3D_z, interestPoint, k):
    #Construct a KD-Tree using the scipy library
    #tree = cKDTree(points3D_xy, leafsize = 2)
    return points3D_z[cKDTree(points3D_xy, leafsize = 2).query(interestPoint, k)[1]]


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ========================================================================================= #
#Function to find the K-nearest-neighbours to the interest point. Using Scikit-Learn. 
#and return ??? (see function's Doc string.)
#@jit(target_backend='cuda')
def k_nearestNeigbours_sklearn(points3D_xy, points3D_z, interestPoint, k, leafSize = 30):
    '''
    INPUT:
    -----
    i. points3D_xy(N x 2 array): xy-Point Cloud data.
    ii. points3D_z(N x 1 array): z-Point Cloud data.
    iii. k(Integer: Default, k = 15 for Normal Vector Estimation, k = 73 for Local Surface Patch Selection.): 
        Area around which nearest points to the 'Interest Points' are obtained.
    iv. leafSize(Integer): See link here for explanation: 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html'

    
    OUTPUT:
    ------
    i. kClossestCoords_xy(N x 2 array): Coordinates of the XY-points neighbourhood within the distance givien by 'k'.
    ii. kClossestCoords_z(N x 1 array): Coordinates of the Z-points neighbourhood within the distance givien by 'k'.
    
    Author: Ekpo (eko@aber.ac.uk)       -        Coded: Tuesday, May 28th, 2019 / Modified: Thursday 17th Feb 2022
    '''
    #from sklearn.neighbors import NearestNeighbors
    
    neigh = cuNearestNeighbors(n_neighbors = k)
    neigh.fit(points3D_xy)
    cuNearestNeighbors(algorithm = 'auto', leaf_size = leafSize) 
    kng = neigh.kneighbors([interestPoint])        #Plug in the 'Interest Point'
    ind = cp.asarray(kng[1][0])    #ind(1 x N array): Contains the indices to neighbouring points returned. N is the size or number of neighbouring points returned.
    kClossestCoords_xy = points3D_xy[ind]    #neighbours(N x 3 array): Coordinates of the points neighbourhood within the distance givien by the radius, 'r'.
    kClossestCoords_z = points3D_z[ind]    #neighbours(N x 3 array): Normal Vectors of the points neighbourhood, i.e neighbours.
    return kClossestCoords_xy, kClossestCoords_z
    
def k_nearestNeigbours_sklearnV2(points3D_xy, points3D_z, interestPoint, k, leafSize = 30):
    '''
    Author: Ekpo (eko@aber.ac.uk)       -        Coded: Thursday 17th Feb 2022
    #from sklearn.neighbors import NearestNeighbors
    '''
    
    neigh = cuNearestNeighbors(n_neighbors = k)
    
    neigh.fit(points3D_xy)
    cuNearestNeighbors(algorithm = 'auto') 
    interestPoint = cp.expand_dims(interestPoint, axis=0)
    #print(interestPoint.shape)
    kng = neigh.kneighbors(interestPoint)
    ind = cp.asarray(kng[1][0])

    return points3D_z[ind]
# ========================================================================================== #
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Function to compute final I values per pixel
def compute_INTENSITYv3(xyValues, points3D_xy, points3D_z, k):
    return [k_nearestNeigbours_sklearnV2(points3D_xy, points3D_z, xy, k).max() for xy in xyValues]
    
# Function to compute final I values per pixel
def compute_INTENSITYv2(xyValues, points3D_xy, points3D_z, k):
    return [k_nearestNeigbourV2(points3D_xy, points3D_z, xy, k).max() for xy in xyValues]

# My custom function to get actual (x,y) axes values from 2D meshgrid
def getMeshgrid_xyValuesV2(xVals, yVals):
    '''
    INPUTS:
    xVals: A 1D array of linearly-spaced values used to compute 2D meshgrid along x-axis. See STEP 3.
    yVals: A 1D array of linearly-spaced values used to compute 2D meshgrid along y-axis. See STEP 3.

    OUTPUT:
    xy_axes: A list of (x,y) tupples, each of which indicates the xy coordinate values of each cell in the 2D meshgrid.
    '''
    return [(i, j) for i in xVals for j in yVals]
#@jit(target_backend='cuda')
def getCustom_xyValuesV2(N, min3d, max3d):
    x_vals = cp.arange(N) # N=512, so i would be a 1D array like so: [0,1,2,3,4...509,510,511] 
    y_vals = cp.arange(N) # N=512, so i would be a 1D array like so: [0,1,2,3,4...509,510,511]

    x_range = max3d[0] - min3d[0]
    y_range = max3d[1] - min3d[1]

    return [((min3d[0] + (i * x_range)/(N-1)), (min3d[1] + (j * y_range)/(N-1))) for i in x_vals for j in y_vals]
    
# FUNCTION to Conver 3D point cloud to 2D intensity image of the 3D's z-coordinate [for all DATABASE 3D models].
#@jit(target_backend='cuda')
def convert3Dxyz_2Dxy_zIntensity_all(database_path, outdir, P, N = 128, k = 10, sample_P_points = False):
    '''
    INPUTS:
        i. database_path (String)
        ii. outdir (String)
        iii. P (Integer)
        iv. N (Integer: Default = 128)
        v. k  (Integer: Default = 10)
        vi. sample_P_points (Boolean: Default = FALSE) 
            If TRUE, this function would automatically sample P points from the  surface of 3D mesh.
            If FALSE, this funtion will automatically use the RAW vertices of the 3D mesh as points, instead.
    OUTPUTS:
        None: Output images are saved to the 'outdir' directory.
    
    '''
    #import os
    #from sklearn.externals import joblib
    #import gc
    
    # Extract the ALL 'filenames.extension' in the directory containing the training files/dataset
    files = os.listdir(database_path)
    
    # COMPUTE 3D-2D CONVERSION FOR EACH 3D MODEL
    for file in files:
        #Get full filepath + filename
        fullpath = database_path + file
        #print("Fullpath:\t", fullpath)
        print("NOW COMPUTING FOR:\t{}".format(file))
        
        # Load 3D mesh
        mesh = trimesh.load(fullpath)
        
        #BOOLEAN CONDITION 1: IF TRUE
        if(sample_P_points):
            # Number of points to sample from surface = P. 

            # Sample P points from mesh surface.
            pointcloud = trimesh.sample.sample_surface(mesh, P)[0]
            #print("Unnormalized pointcloud:\n", pointcloud)
            print("Number of Points (P):\t", P)
            print("pointcloud:\n", pointcloud[:5])
            
        #BOOLEAN CONDITION 2: IF FALSE
        else:
            pointcloud = mesh.vertices
            
            pointcloud = cp.asarray(pointcloud)
            print(pointcloud.shape)
            
            P = pointcloud.shape[0]

            #print("Unnormalized pointcloud:\n", pointcloud.shape)
            #print("Unnormalized pointcloud Center:\t", np.mean(pointcloud, axis=0))
            #print("Unnormalized pointcloud Minimum:\t", np.min(pointcloud, axis=0))
            #print("Unnormalized pointcloud Maximum:\t", np.max(pointcloud, axis=0))

            print("Number of Vertices (P):\t", P)
        
        print("\nWHICH P?", P, "\n")
        #pointcloud = pointcloud[:10000,:]
        # '''FLIP [x,y,z]'''
        xx = pointcloud[:, 0]
        yy = pointcloud[:, 2]
        zz = pointcloud[:, 1]
        pointcloud = cp.column_stack((xx, yy, zz))
        #print("\nnpointcloudDDD:\n", pointcloud[:5])
        
        min_xyz = cp.min(pointcloud, axis = 0)
        max_xyz = cp.max(pointcloud, axis = 0)
        print("Min pointcloud coords:\t", min_xyz)
        print("Max pointcloud coords:\t", max_xyz)

        xyzRange = max_xyz - min_xyz
        print("xyzRange.max():\t", xyzRange.max())
        print("xyzRange.max(0):\t", xyzRange.max(0))
        print("xyzRange[:2].max():\t", xyzRange[:2].max())
        print("xyzRange[:2].max(0):\t", xyzRange[:2].max(0))

        # UPDATED WITH LIST COMPREHENSION
        cvv = [i for i in range(0, len(xyzRange)) if(xyzRange[i] == xyzRange[:2].max())][0]
        print("cvv:\t", cvv)

        # In actual sense, N = M = 512. The values below are for testing purposes only.

        xVals = cp.linspace(min_xyz[cvv], max_xyz[cvv], N)
        yVals = cp.linspace(min_xyz[cvv], max_xyz[cvv], N)

        # Create 2D meshgrid
        x, y = cp.meshgrid(xVals, yVals)


        # Get all the (x,y)-axes values of the 2D meshgrid created in STEP 4
        xyValues = getMeshgrid_xyValuesV2(xVals, yVals)

        print('xyValues Size: ', cp.asarray(xyValues).shape)


        # Ensure that xyValues is a proper Numpy array!
        xyValues = cp.asarray(xyValues)
        #print(xyValues.shape)
        print("\nxyValues Data:\n", xyValues)
        #xyValues = xyValues[:10000,:] 
        # Select only the first two [x,y] columns of the three [x,y,z] columns in point cloud.
        print(pointcloud[:, 2:])
        pointcloud_xy = pointcloud[:, :2]
        print("pointcloud_xy.shape: ", pointcloud_xy.shape)
        min2d = cp.min(xyValues, 0)
        max2d = cp.max(xyValues, 0)
        min3d = cp.min(pointcloud_xy, 0)
        max3d = cp.max(pointcloud_xy, 0)
        print("\nmin2d:\t", min2d, "\nmax2d:\t", max2d, "\n\nmin3d:\t", min3d, "\nmax3d:\t", max3d)

        # Call INTENSITY computation function
        s1 =  time.time()
        #I2 = normalize1Darrayx(compute_INTENSITYv2(xyValues, pointcloud_xy, pointcloud[:, 2], k))
        I2 = normalize1Darrayx2(compute_INTENSITYv3(xyValues, pointcloud_xy, pointcloud[:, 2], k))
        s2 =  time.time()
        print("Duration:\t", s2-s1, " Seconds")
        
        # Inspect the computed I values
        #print("I2 values:\n", I2)
        print("I2 values SHAPE:\n", cp.asarray(I2).shape)

        intensity2Dv2 = cp.reshape(I2, (N, N))
        print("intensity2Dv2 SHAPE:\n", intensity2Dv2.shape)
        print("\nintensity2Dv2 VALUES:\n", intensity2Dv2)
        
        filename1a = outdir + file[:-4] + "_P{}_N{}_K{}pltA.png".format(P, N, k)
        filename2a = outdir + file[:-4] + "_P{}_N{}_K{}pilA.png".format(P, N, k)
        
        # FIRST, let's try to use Matplotlib's plotting library - For visualizing the INTENSITY values over a 2D grid
        '''
        We will plot a 2D contour surface at equaly spaced intervals of the I values on the contour.
        '''
        x = cp.asnumpy(x)
        y = cp.asnumpy(y)
        intensity2Dv2 = cp.asnumpy(intensity2Dv2)
        plt.figure()
        plt.contourf(x, y, intensity2Dv2, levels = 100, cmap = plt.cm.coolwarm)
        plt.colorbar()
        plt.savefig(filename1a)
        plt.close()
        #print(intensity2Dv2.shape)
        # Creates PIL image
        img = Image.fromarray(cp.uint8(intensity2Dv2 * 255) , 'L')
        img.save(filename2a)
        break
        '''
        xy_vals = getCustom_xyValuesV2(N, max3d, min3d)

        I3 = normalize1Darrayx(compute_INTENSITYv2(xy_vals, pointcloud_xy, pointcloud[:, 2], k))
        intensity2Dv3 = np.reshape(I3, (N,N))

        filename1b = outdir + file[:-4] + "_P{}_N{}_K{}pltB.png".format(P, N, k)
        filename2b = outdir + file[:-4] + "_P{}_N{}_K{}pilB.png".format(P, N, k)

        plt.figure()
        np_array = plt.contourf(x, y, intensity2Dv3, levels = 100, cmap = plt.cm.coolwarm)
        plt.colorbar()
        plt.savefig(filename1b)

        # Creates PIL image
        img = Image.fromarray(np.uint8(intensity2Dv3 * 255) , 'L')
        img.save(filename2b)
        '''
             
        
# ===================================================================================================================== # 
# ## CALL FUNCTION TO COMPUTE 3D-2D FOR ALL DATABASE MODELS

inpath = "Swansea/"
outpath = "Out/"
P = 15257
#P = 20000
N = 512
#N=256
k = 1
sample_P_points = False
# ============================================ RUN CODES HERE ========================================================= # 
# Call funtion and compute 2D intensity image for all 3D meshes.
start_time = time.time()

convert3Dxyz_2Dxy_zIntensity_all(inpath, outpath, P, N, k, sample_P_points)

end_time = time.time()
print("time takes: {}".format(end_time - start_time))
# ============================================ RUN CODES HERE ========================================================= # 
