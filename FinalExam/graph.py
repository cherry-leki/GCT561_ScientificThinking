import numpy as np
from matplotlib import pyplot as plt

h = np.array(range(10001))
h = h / 10000
y = list()

max = 0
sum = 0
for i in range(0, 10001):
    tmp = 464 * (h[i] ** 4) * ((1-h[i]) ** 6)
    y.append(tmp)
    if i <= 399:
        sum += tmp
    if max < tmp:
        max = tmp

print(max , ", " , sum)

h2 = np.array(range(10001))
h2 = h2 / 10000
y2 = list()

max = 0
sum = 0
for i in range(0, 10001):
    tmp = 252 * (h2[i] ** 3) * ((1 - h2[i]) ** 7)
    y2.append(tmp)

    if i >= 0.3:
        sum + tmp

    if max < tmp:
        max = tmp

print(max, ", " , sum)

plt.plot(h, y)
plt.plot(h2, y2)
plt.xlim(0, 1)
plt.show()