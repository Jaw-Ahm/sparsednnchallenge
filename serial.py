import numpy as np
import scipy.sparse as sp
from time import time

inputfile = "C:/GraduateResearch/SDNN DATA/data/MNIST/sparse-images-"
categoryfile = "C:/GraduateResearch/SDNN DATA/data/DNN/neuron"
layerfile = "C:/GraduateResearch/SDNN DATA/data/DNN/neuron"

maxlayers = [120, 480, 1920]
neuralnetbias = [-0.3, -0.35, -0.4, -0.45]
nneurons = [1024, 4096, 16384, 65536]

def inferenceReLUvec(w, b, y):
    """
    :param w: DNN weights
    :param b: constant bias
    :param y: input vector(s)
    :return: scores
    """
    ymax = 32

    for i, weight in enumerate(w):
        z = y * weight
        z.data += b
        y = z
        y.data[y.data < 0] = 0
        y.data[y.data > ymax] = ymax
    return y



def readtriples(f, s):
    """
    Reads triples into csr_matrix (sparse matrix) format
    :param f: File name
    :param s: Size of dense matrix
    :return: csr_matrix
    """
    m, n = s
    A = np.genfromtxt(fname=f, delimiter='\t', dtype=np.float32).transpose()
    # need to subtract 1 to have first element at index at 0
    A[0:2] = A[0:2] - 1
    A = sp.csr_matrix((A[2], (A[0], A[1])), shape=(m,n))
    return A


def main():
    # loop over each DNN
    for i, n in enumerate(nneurons):

        # read MNIST data into csr_matrix (sparse matrix)
        featurevectors = readtriples(f=f"{inputfile}{n}.tsv", s=(60000,n))
        nfeaturevectors = featurevectors.shape[0]
        # read layers
        for m in maxlayers:

            # read true categories
            truecats = np.zeros(60000)
            cats = np.genfromtxt(fname=f"{categoryfile}{n}/neuron{n}-l{m}-categories.tsv", delimiter='\t', dtype=int).transpose()
            cats = cats - 1
            truecats[cats] = 1

            dnnedges = 0
            layers = []
            # bias = []
            start = time()

            for l in range(1, m+1):
                layers.append(readtriples(f=f"{layerfile}{n}/n{n}-l{l}.tsv",s=(n, n)))
                dnnedges += layers[l-1].getnnz()
                # bias.append(sp.csr_matrix(np.multiply(np.ones((1, n)),  neuralnetbias[i])))

            readlayertime = time() - start
            readlayerrate = dnnedges/readlayertime

            print(f"DNN neurons/layer: {n}, layers: {l}, edges: {dnnedges}")
            print(f"Read time (sec): {readlayertime:.2f}, read rate (edges/sec): {readlayerrate:e}")

            start = time()
            scores = inferenceReLUvec(layers, neuralnetbias[i], featurevectors)
            challengeruntime = time() - start
            challengerunrate = (nfeaturevectors * dnnedges) / challengeruntime

            print(f"Run time (sec): {challengeruntime:.2f}, run rate (edges/sec): {challengerunrate:e}")

            # compute categories from scores
            categories = np.zeros(60000)
            categories[np.where(scores.sum(axis=1) != 0)[0]] = 1

            categorydiff = sum(categories != truecats)

            print(f"Errors: {categorydiff}")

if __name__ == "__main__":
    main()