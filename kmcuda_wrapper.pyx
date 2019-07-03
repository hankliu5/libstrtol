#cython: language_level=3
from libc.stdint cimport uint32_t, uint16_t, int32_t
from libc.stdlib cimport calloc, free

import time
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "kmcuda.h":
    ctypedef enum KMCUDAResult:
        kmcudaSuccess = 0
        kmcudaInvalidArguments
        kmcudaNoSuchDevice
        kmcudaMemoryAllocationFailure
        kmcudaRuntimeError
        kmcudaMemoryCopyError

    ctypedef enum KMCUDAInitMethod:
        kmcudaInitMethodRandom = 0
        kmcudaInitMethodPlusPlus
        kmcudaInitMethodAFKMC2
        kmcudaInitMethodImport
    
    ctypedef enum KMCUDADistanceMetric:
        kmcudaDistanceMetricL2
        kmcudaDistanceMetricCosine 

    KMCUDAResult kmeans_cuda(
    KMCUDAInitMethod init, const void *init_params, float tolerance, float yinyang_t,
    KMCUDADistanceMetric metric, uint32_t samples_size, uint16_t features_size,
    uint32_t clusters_size, uint32_t seed, uint32_t device, int32_t device_ptrs,
    int32_t fp16x2, int32_t verbosity, const float *samples, float *centroids,
    uint32_t *assignments, float *average_distance);


def cython_kmeans_cuda(
    float tolerance, float yinyang_t,
    uint32_t clusters_size, uint32_t seed, uint32_t device, 
    int32_t device_ptrs, int32_t fp16x2, int32_t verbosity, np.ndarray[float, ndim=2, mode="c"] samples not None):
    
    cdef int samples_size, features_size
    samples_size, features_size = samples.shape[0], samples.shape[1]
    cdef float* centroids = <float *> calloc(clusters_size * features_size, sizeof(float));
    cdef uint32_t* assignments = <uint32_t *> calloc(samples_size, sizeof(uint32_t));
    cdef float average_distance;
    
    start = time.time()
    print(samples)
    cdef KMCUDAResult result = kmeans_cuda(
        kmcudaInitMethodPlusPlus, NULL, tolerance, yinyang_t, kmcudaDistanceMetricL2, samples_size, 
        features_size, clusters_size, seed, device, device_ptrs, fp16x2, 
        verbosity, &samples[0,0], centroids, assignments, &average_distance);
    end = time.time()
    print("kmeans elapsed time: {}".format(end - start))

    free(centroids);
    free(assignments);

    return <int> result