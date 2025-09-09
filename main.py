import random
from math import isnan

import numpy as np
import scipy
from scipy.stats.qmc import Halton
from scipy.stats import linregress
import matplotlib.pyplot as plt


trueGamma = np.euler_gamma

####################################### QUESTION 1 #######################################

def getSample(size, c = 1.0, alpha = 1):
    U = np.random.uniform(0, 1, size)
    X = np.ceil(c / (U ** alpha))
    return X.astype(int)

def p(k, c = 1.0, alpha = 1):
    return (1 / k) - np.log(1 + 1 / k) + (1 - np.log(2)) * (c / (k - 1) ** alpha - c / k ** alpha)  #We don't divide by gamma because later we multiply by gamma and want to avoid numerical error

def q(k, c = 1.0, alpha = 1):
    return c / (k - 1)**alpha - c/ k**alpha

def importanceSampling(size, c = 1.0, alpha = 1):
    samples = getSample(size, c, alpha)
    weights = p(samples, c, alpha) / q(samples, c, alpha)
    return np.mean(weights)

####################################### QUESTION 2 #######################################

def integrand(x):
    return -np.log(-np.log(x))

def monteInt(size):
    samples = np.random.uniform(0, 1, size)
    return np.mean(integrand(samples))

def createPartition(size):
    indices = np.arange(1, size)
    alpha = np.sqrt(size)
    inner_points = (1 + np.tanh(alpha * (indices / size - 0.5))) / 2
    return np.concatenate(([0], inner_points, [1]))

def plotPartition(partition):
    plt.figure(figsize=(8, 4))
    plt.hlines(1, 0, 1, color='gray', linestyles='dashed')
    for x in partition:
        plt.vlines(x, 0, 1, color='blue', linestyle='dotted')
    plt.scatter(partition, np.ones_like(partition), color='red', label='Partition Points')
    plt.xlabel("x")
    plt.title("Visualization of Partition Intervals")
    plt.legend()
    plt.show()


def stratifiedMonte(size):
    partition = createPartition(size)
    totalSamples = size * 10
    integral = 0

    for k in range(len(partition) - 1):
        a_k = partition[k]
        a_k1 = partition[k + 1]
        length = a_k1 - a_k
        size = int(totalSamples * length)

        samples = np.random.uniform(a_k, a_k1, size + 1)
        integral += np.mean(integrand(samples)) * length
    return integral


def quasiMonte(size):
    haltonSeq = Halton(1).random(n = size).flatten()
    return np.mean(integrand(haltonSeq))

####################################### QUESTION 3 #######################################

def cvMonteInt(size):
    samples = np.random.uniform(0, 1, size)
    cv = np.log(1 - samples)
    approx = integrand(samples)

    cov = np.cov(approx, cv)[0, 1]
    if isnan(cov):
        cov = 0
    var = np.var(cv)
    beta = cov/var
    if isnan(beta):
        beta = 0
    return np.mean(approx - beta*(cv + 1)) #Avg of cv is -1

def cvQuasiMonte(size):
    haltonSeq = Halton(1).random(n=size).flatten()
    cv = np.log(1 - haltonSeq)
    approx = integrand(haltonSeq)

    cov = np.cov(approx, cv)[0, 1]
    var = np.var(cv)
    beta = cov / var
    if isnan(beta):
        beta = 0
    return np.mean(approx - beta * (cv + 1))  # Avg of cv is -1

####################################### QUESTION 4 #######################################

def gammaSeq(k):
    x = np.arange(1, k + 1)
    inv = 1/x
    logs = np.log(1 + inv)
    return inv - logs

def geoProbaSeq(p, k):
    x = np.arange(1, k + 1)
    return (1 - p) ** (x-1) #P(R >= k)

def geoSum(size, p = .35):
    samples = np.random.geometric(p, size)
    approx = []
    for r in samples:
        if r == 0:
            continue
        approx += [sum(gammaSeq(r)/geoProbaSeq(p, r))]
    return np.mean(approx)

def logRange(start = 1, stop = 100000, ptsPerDecade=3): #Evenly spaced values for log log plot
    logStart = np.log10(start)
    logStop = np.log10(stop)

    values = np.logspace(logStart, logStop, num=int((logStop - logStart) * ptsPerDecade) + 1)

    return np.unique(np.round(values).astype(int)).tolist()




def compareApprox(samplesRange, sampleSize, methods, labels):
    plt.figure(figsize=(10, 6))
    orders = []

    for method, label in zip(methods, labels):
        approxErrors = []
        approxErrorStdDev = []

        for sample in samplesRange:
            sampleError = []
            for _ in range(sampleSize):
                approxError = abs(trueGamma - method(sample))
                sampleError.append(approxError)

            approxErrors.append(np.mean(sampleError))
            approxErrorStdDev.append(np.std(sampleError))

        plt.errorbar(samplesRange, approxErrors, yerr=approxErrorStdDev, fmt='o-', capsize=5, label=label)

        log_samples = np.log(samplesRange)
        log_errors = np.log(approxErrors)
        slope, intercept, r_value, p_value, std_err = linregress(log_samples, log_errors)
        orders.append(slope)
        print(f"Method '{label}' has order: {slope:.3f} (R² = {r_value ** 2:.3f})")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel(r'$|\gamma - \text{approx}|$')
    plt.title(r'Comparison of Approximation Errors')
    plt.legend()
    plt.grid(True)
    plt.show()

    return orders



samplesRange = logRange()
sampleSize = 30
methods = [importanceSampling, monteInt, cvMonteInt, stratifiedMonte, quasiMonte, cvQuasiMonte, geoSum]
labels =  ['Importance Sampling', 'Standard Monte Carlo', 'Standard Monte Carlo With Control Variates',
           'Stratified Monte Carlo', 'Quasi Monte Carlo', 'Quasi Monte Carlo With Control Variates', 'Truncated Sum']

compareApprox(samplesRange, sampleSize, methods, labels) #If execution is too long reduce sample size and pts per decile
