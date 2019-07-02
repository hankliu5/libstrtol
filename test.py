import hello
import numpy
import pandas
import time
import sys
import cython_wrapper


if len(sys.argv) < 2:
    print("usage {} [input data]".format(sys.argv[0]))
    exit(1)

read_time = 0
strtol_transform_time = 0

start = time.time()
f = open(sys.argv[1], 'r')
s = f.read()
end = time.time()
read_time = end - start

start = time.time()
a = hello.deserialize(s)
end = time.time()
strtol_transform_time = end - start
f.close()
print("read time: {}".format(read_time))
print("c api elapsed time: {}".format(strtol_transform_time))

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

start = time.time()
f = open(sys.argv[1], 'r')
s = f.readline().rstrip()
row, _ = s.split()
end = time.time()
read_time = end - start

start = time.time()
c = pandas.read_csv(f, sep=' ', header=None, nrows=int(row)).to_numpy()
end = time.time()
strtol_transform_time = end - start

f.close()

print("read time: {}".format(read_time))
print("pandas elapsed time: {}".format(strtol_transform_time))

print('same: {}'.format(numpy.array_equal(a, b) and numpy.array_equal(a, c)))