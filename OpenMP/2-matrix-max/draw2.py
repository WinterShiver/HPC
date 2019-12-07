# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import math

# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
# matplotlib.rcParams['font.serif'] = ['KaiTi']


dims = ["$2^{"+str(n)+"}$" for n in range(8, 13)]
times1 = [0.0002887, 0.0007073, 0.0038686, 0.0134514, 0.0540589]
times1 = [math.log(elem) for elem in times1]
times2 = [0.0002335, 0.0007808, 0.0018217, 0.0074485, 0.0319176]
times2 = [math.log(elem) for elem in times2]

x_label = "Matrix Dimension"
y_label = "Runtime"
plt.plot(dims, times1, 'purple', label = "1 Thread")
plt.plot(dims, times2, 'green', label = "2 Threads")
loc, labels = plt.yticks()
print(loc, labels)
plt.yticks([elem + 0.3 for elem in list(loc)], [str('$10^{' + str(int(elem)) + '}s$') for elem in loc], rotation=0)
plt.title("Runtime of Finding Max Elem in Matrix\n(Implemented by OpenMP)")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend()
plt.grid()
plt.savefig("2.png")
plt.clf()