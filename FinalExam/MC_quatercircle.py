import random
from matplotlib import pyplot as plt
import seaborn as sns
import itertools
import math

n = 1000
prob = list()

def testMC(n):
    k = 0
    for i in range(0, n):
        x = random.uniform(0, 1.0)
        y = random.uniform(0, 1.0)
        if pow(x, 2) + pow(y, 2) <= 1:
            k = k + 1
    return float(k)/float(n)

def kMC():
    result = list()
    max = 0
    num = 0
    for i in range(0, 100):
        c = math.factorial(100) / (math.factorial(i) * math.factorial(100-i))
        p = c * ((math.pi/4)**i) * ((1-(math.pi/4))**(100-i))
        result.append(p)
        if max < p:
            max = p
            num = i

    return result, num


if __name__=="__main__":
    for i in range(1, n+1):
        prob.append(testMC(i))

    # plt.plot(list(range(1, 101)), prob)
    # plt.show()

    # print(prob[n-1])
    # sns.set(color_codes=True)
    # sns.distplot(prob)

    r, num = kMC()

    print(r[num], num)
    sum = 0
    for i in range(0, 25):
        sum += r[i]
    print(sum)
    sum = 0
    for i in range(25, 50):
        sum += r[i]
    print(sum)
    sum = 0
    for i in range(50, 75):
        sum += r[i]
    print(sum)
    sum = 0
    for i in range(75, 90):
        sum += r[i]
    print(sum)
    sum = 0
    for i in range(90, 101):
        if i == 100:
            break
        sum += r[i]
    print(sum)
    sum = 0



    plt.plot(list(range(1,101)),r)
    plt.xlim(1, 100)
    plt.show()