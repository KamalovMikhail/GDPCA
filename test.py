from utils import *
from GDPCA import GDPCA
from sklearn.metrics.pairwise import  cosine_similarity , rbf_kernel
from sklearn import metrics as m
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#####################################################
#### Example for mnist dataset without default adjacency matrix
#### nl - the number of labelld nodes from each class
#### n_samples - the numebr of objects
#### train_idx - the list of object indexes wit lalbels
#####################################################
nl = 20
digits = datasets.load_digits()
n_samples = len(digits.images)
features = digits.images.reshape((n_samples, -1))
#### Select indexes of laballed objects from each class for generation of matrix y_train
unique_y = np.unique(digits.target)
train_idx = np.array([])
for k in unique_y:
    train_idx = np.concatenate([train_idx,
                                np.random.choice(np.where(digits.target == k)[0],
                                                 size=nl,
                                                 replace=False)])
train_idx = [int(i) for i in train_idx]
#### Generate y_train matrix n x k, where n  is the number of objects and k is a number of classes
y_train = np.zeros((n_samples, len(unique_y)))
for i in train_idx:
    y_train[i, digits.target[train_idx]] = 1

#####################################################
#### Example for citation networks with deafult adjacency matrix
#####################################################
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
#Af = adj.toarray()
Af = generate_adj(np.array(features))
#### Experiments with another similarity
#MMCos = cosine_similarity(features.toarray()) / (d-1)
#MMrbf = rbf_kernel(features.toarray())/(d - 1)

predicts = GDPCA(features, Af, y_train, delta=1,
                 sigma=1, alpha=0.9, svd_error=1e-03,
                 iter_=10, tol=1e-03)
print('accuracy',m.accuracy_score(digits.target,
                                  np.argmax(np.array(predicts[0]), axis=-1)))
print('time', predicts[1])
#### Results for citation networks
#print('accuracy',m.accuracy_score(np.argmax(y_test[test_mask], axis=-1),
#                                  np.argmax(np.array(predicts[0])[test_mask], axis=-1) ))
#print('time', predicts[1])
