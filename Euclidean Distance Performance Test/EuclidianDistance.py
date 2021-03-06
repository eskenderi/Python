import numpy as np
import time
import matplotlib.pyplot as plt


def euclidian_distance_loop(X):
    N = np.zeros([len(X), len(X)])
    xxT=X.dot(X.transpose())

    for i in range(len(N)):
        for j in range(i+1, len(N)):
            N[i][j] = pow(xxT[i][i] + xxT[j][j] - 2 * xxT[i][j], 0.5)
            N[j][i] = N[i][j]
    return N


def euclidian_distance_fast(X):
    N = np.zeros([len(X), len(X)])

    for row in range(len(N)):
        N[row] = np.sqrt(np.sum(np.power(np.subtract(X, X[row]), 2), axis=1))
    return N

np.random.seed(13)
features = range(30, 130, 10)

res_loop = np.zeros([10, len(features)])
res_fast = np.zeros([10, len(features)])

cnt = 0

for cols in features:
    rows = cols * 10

    print "Dimensions: ", rows, cols

    for i in range(10):
        X = np.random.randint(0, 20, [rows, cols])

        start = time.time()
        euclid_dist_loop = euclidian_distance_loop(X)
        finish = time.time()
        res_loop[i, cnt] = finish - start

        start = time.time()
        euclid_dist_fast = euclidian_distance_fast(X)
        finish = time.time()
        res_fast[i, cnt] = finish - start

        assert np.isclose(euclid_dist_loop[1][2], euclid_dist_fast[1][2], atol=1.e-5)
    cnt += 1


mean_loop = np.mean(res_loop, axis=0)
mean_fast = np.mean(res_fast, axis=0)

std_loop = np.std(res_loop, axis=0)
std_fast = np.std(res_fast, axis=0)

plt.errorbar(features, mean_loop[0:len(features)], yerr=std_loop[0:len(features)], color='red', label = 'Loop Solution')
plt.errorbar(features, mean_fast[0:len(features)], yerr=std_fast[0:len(features)], color='blue', label = 'Fast Solution')
plt.xlabel('Number of cols')
plt.ylabel('Running Time (Seconds)')
plt.legend()
plt.savefig('CompareEuclidDistance.pdf')
plt.show()