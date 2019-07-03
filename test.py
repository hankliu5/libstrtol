import py_deserialize
import numpy
import pandas
import time
import sys
import cython_wrapper
import kmcuda_wrapper

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

start = time.time()
f = open(sys.argv[1], 'rb')
s = f.read()
end = time.time()
read_time = end - start

start = time.time()
b = cython_wrapper.cython_deserialize(s)
end = time.time()
strtol_transform_time = end - start

f.close()
print("read time: {}".format(read_time))
print("cython elapsed time: {}".format(strtol_transform_time))

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

start = time.time()
b = b.astype('float32')
end = time.time()
print("float transform elapsed time: {}".format(end - start))

start = time.time()
kmcuda_wrapper.cython_kmeans_cuda(0.01, 0.1, 5, 0x5, 0, -1, 0, 0, b);
end = time.time()

print("kmeans elapsed time: {}".format(end - start))
