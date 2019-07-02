import hello
import numpy
import pandas
import time
import sys
import cython_wrapper


if len(sys.argv) < 2:
    print("usage {} [input data]".format(sys.argv[0]))
    exit(1)

start = time.time()
f = open(sys.argv[1], 'r')
s = f.read()
a = hello.deserialize(s)
f.close()
end = time.time()
print("c api elapsed time: {}".format(end-start))

start = time.time()
f = open(sys.argv[1], 'rb')
s = f.read()
b = cython_wrapper.cython_deserialize(s)
f.close()
end = time.time()
print("cython elapsed time: {}".format(end-start))

start = time.time()
f = open(sys.argv[1], 'r')
s = f.readline().rstrip()
row, _ = s.split()
c = pandas.read_csv(f, sep=' ', header=None, nrows=int(row)).to_numpy()
f.close()
end = time.time()
print("pandas elapsed time: {}".format(end-start))

print('same: {}'.format(numpy.array_equal(a, b) and numpy.array_equal(a, c)))