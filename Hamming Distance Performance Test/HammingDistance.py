import numpy as np
import matplotlib.pyplot as plt
import time


def hamming_dist_loop(X):
    N = np.zeros([len(X), len(X)])

    for i in range(len(X)):
        for j in range(i+1, len(X)):

            for c in range(len(X[0])):
                if X[i][c] != X[j][c]:
                    N[i][j] += 1
            N[j][i] = N[i][j]
    return N

def hamming_dist_fast(X):
    N = np.zeros([len(X), len(X)])

    for row in range(len(X)):
        N[row] = np.sum(np.subtract(X,X[row])!=0,axis=1)

    return N

np.random.seed(33)
features = range(20, 50, 2)

res_loop = np.zeros([10, len(features)])
res_fast = np.zeros([10,len(features)])

cnt = 0

for cols in features:
    rows = cols*10

    print "Dimensions: ", rows, cols

    for i in range(10):
        X = np.random.randint(0, 2, [rows, cols])

        start = time.time()
        hamm_dist_loop = hamming_dist_loop(X)
        finish = time.time()
        res_loop[i, cnt] = finish - start

        start = time.time()
        hamm_dist_fast = hamming_dist_fast(X)
        finish = time.time()
        res_fast[i, cnt] = finish - start

        assert np.isclose(np.sum(hamm_dist_loop), np.sum(hamm_dist_fast), atol=1.e-5)
    cnt += 1

mean_loop = np.mean(res_loop, axis=0)
mean_fast = np.mean(res_fast, axis=0)

std_loop = np.std(res_loop, axis=0)
std_fast = np.std(res_fast, axis=0)

plt.errorbar(features, mean_loop[0:len(features)], yerr=std_loop[0:len(features)], color='red', label='Loop Solution')
plt.errorbar(features, mean_fast[0:len(features)], yerr=std_fast[0:len(features)], color='blue', label='Fast Solution')
plt.xlabel('Number of Colons')
plt.ylabel('Running Time (Seconds)')
plt.legend()
plt.savefig('CompareHammingtonDist.pdf')
plt.show()
