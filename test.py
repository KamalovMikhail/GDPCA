from utils import *
from GDPCA import GDPCA
from sklearn.metrics.pairwise import  cosine_similarity , rbf_kernel
from sklearn import metrics as m
from sklearn.neighbors import NearestNeighbors


def generate_adj(X, metric='minkowski', n_neighbours=25):
    ##############################################################################
    ### generate_adj - The utility method for generation of Adjacency matrix.
    ### Input:
    ### X - is a matrix (n x d) of node features, where d is the number of features (np.ndarray);
    ### metric - is a metric of distance between nodes for NearestNeighbors algorithm (string);
    ### https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    ### n_neighbours - is the number of neighbours for NearestNeighbors algorithm (int);
    ### Output:
    ### A - is a symmetrixc (adjacency/similarity) matrix (n x n), where n is the number of nodes (np.ndarray);
    ##############################################################################
    nnodes = X.shape[0]
    A = np.zeros((nnodes, nnodes))
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, metric=metric).fit(X)
    ind_ =  0
    for x in X:
        _, indices = nbrs.kneighbors(x)
        A[ind_, indices] = 1
        A[indices, ind_ ] = 1
        ind_ += 1
    return A

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
#Af = adj.toarray()


Af = generate_adj(features)
nnodes = Af.shape[0]
d = features.toarray().shape[1]
# experiments with another similarity
#MMCos = cosine_similarity(features.toarray()) / (d-1)
#MMrbf = rbf_kernel(features.toarray())/(d - 1)

predicts = GDPCA(features.toarray(), Af, y_train, delta=1,
                 sigma=1, alpha=0.9, svd_error=1e-03,
                 iter_=10, tol=1e-03)

print('accuracy',m.accuracy_score(np.argmax(y_test[test_mask], axis=-1),
                                  np.argmax(np.array(predicts[0])[test_mask], axis=-1) ))
print('time', predicts[1])