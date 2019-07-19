import py_deserialize
import numpy
import pandas
import time
import sys
import cython_wrapper
import kmcuda_wrapper
import libKMCUDA

if len(sys.argv) < 2:
    print("usage {} [input data]".format(sys.argv[0]))
    exit(1)

# read_time = 0
# strtol_transform_time = 0

# start = time.time()
# f = open(sys.argv[1], 'r')
# s = f.read()
# end = time.time()
# read_time = end - start

# start = time.time()
# a = py_deserialize.deserialize(s)
# end = time.time()
# strtol_transform_time = end - start
# f.close()
# print("read time: {}".format(read_time))
# print("c api elapsed time: {}".format(strtol_transform_time))

with open(sys.argv[1], 'rb') as f:
    start = time.time()
    s = f.read()
    duration = time.time() - start
    print("read time: {}".format(duration))

start = time.time()
a = cython_wrapper.cython_deserialize(s)
duration = time.time() - start
print("cython elapsed time: {}".format(duration))

start = time.time()
b = cython_wrapper.cython_deserialize2(s)
duration = time.time() - start
print("customized cython elapsed time: {}".format(duration))

start = time.time()
c = cython_wrapper.cython_deserialize3(s)
duration = time.time() - start
print("customized cython without wrapper elapsed time: {}".format(duration))

print("a, b are the same: {}".format(numpy.array_equal(a, b)))
print("b, c are the same: {}".format(numpy.array_equal(b, c)))

# start = time.time()
# f = open(sys.argv[1], 'r')
# s = f.readline().rstrip()
# row, _ = s.split()
# end = time.time()
# read_time = end - start

# start = time.time()
# c = pandas.read_csv(f, sep=' ', header=None, nrows=int(row)).to_numpy()
# end = time.time()
# strtol_transform_time = end - start

# f.close()

# print("read time: {}".format(read_time))
# print("pandas elapsed time: {}".format(strtol_transform_time))

# print('same: {}'.format(numpy.array_equal(a, b) and numpy.array_equal(a, c)))

# start = time.time()
# b = b.astype('float32')
# duration = time.time() - start
# print("float transform elapsed time: {}".format(duration))

# start = time.time()
# libKMCUDA.kmeans_cuda(samples=b, clusters=5, tolerance=0.01, yinyang_t=0.1, metric="L2", device=0, verbosity=0, seed=0x5);
# # kmcuda_wrapper.cython_kmeans_cuda(0.01, 0.1, 5, 0x5, 0, -1, 0, 0, b);
# duration = time.time() - start
# print("kmeans elapsed time: {}".format(duration))
