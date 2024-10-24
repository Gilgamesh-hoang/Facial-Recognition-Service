'''
    Testing out SGDClassifier's parameters, especially how to use 'fit' and
    'partial_fit'
'''


import numpy as np
import scipy
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # Example 2: Generate fake data (complex, large)
    print( "\nExample 2")
    np.random.seed(1234)
    nsamples = 100
    nfeatures = 1000
    numOnesInX = np.int32(0.01 * nsamples * nfeatures)  # 1000  # <type 'int'>
    nlabels = 4
    row_array = np.random.randint(0, nsamples, size=numOnesInX)
    # row_array is: [47 83 38 53 ... ] <type 'numpy.ndarray'>  (1000L,)
    col_array = np.random.randint(0, nsamples, size=numOnesInX)
    # col_array is: [82 44 48 26 ... ] <type 'numpy.ndarray'>  (1000L,)
    data = np.ones_like(row_array)
    # data is: [1 1 1 1 1 1 ... ] <type 'numpy.ndarray'>  (1000L,)

    # data is converted into 'Compressed Sparse Rows representation'
    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    X = scipy.sparse.csc_matrix((data, (row_array, col_array)),
                                shape=(nsamples, nfeatures))
    # X is: <class 'scipy.sparse.csc.csc_matrix'>
    # (5, 0)    1
    # (11,0)    1
    y = np.random.randint(0, 2, size=[nsamples, nlabels])
    # Y is: <type 'numpy.ndarray'  (100L, 4L)
    #    [[0 1 1 0]
    #     [1 1 0 1]
    # ... [1 0 0 0] ]

    # partial_fit with partial_fit(X, y[, classes, sample_weight])
    clf_2 = SGDClassifier(alpha=.0001, loss='perceptron', penalty='l2', n_jobs=-1,
                          #shuffle=True, n_iter=10,
                          verbose=1)
    print( "Running Partial Fit 1")
    clf_2.fit(X[:99, :], y[:99, 0])
    pred = clf_2.predict(X[:10, :])
    print( "Prediction is: ", pred)  # [1 1 0 0 1 1 0 0 1 1]
    print( "Accuracy Score is: ", accuracy_score(pred, y[:10, 0]) ) # gives 0.9

    print( "Running Partial Fit 2")
    clf_2.partial_fit(X[:99, :], y[:99, 0])  # fit again on same data!
    pred = clf_2.predict(X[:10, :])
    print( "Prediction is: ", pred ) # [0 1 0 0 0 1 0 0 1 0]
    print( "Accuracy Score is: ", accuracy_score(pred, y[:10, 0]))  # now 0.8