import pandas as pd 
import numpy as np 
import os 
from ase import Atoms 
import json 
from dscribe.descriptors import MBTR
import pickle 
### Define filename of training and test data
train_filename = "train.json"
test_filename = "test.json"
### 


cwd = __file__.split("/")[1:-1]

train_dir = ""
test_dir = ""
for dir in cwd:
    train_dir += "/" + dir 
    test_dir += "/" + dir 

train_dir += "/" + train_filename
test_dir += "/" + test_filename
print("") 
print(train_dir)
print("")
print(test_dir)
print("")



train_data = pd.DataFrame(json.load(open(train_dir, "rb")))
test_data = pd.DataFrame(json.load(open(test_dir, "rb")))

train_data.atoms = train_data.atoms.apply(lambda x: Atoms(**x))
test_data.atoms = test_data.atoms.apply(lambda x: Atoms(**x))


# Determine species
species = []
for atom in train_data.atoms:
    species = list(set(species+atom.get_chemical_symbols()))


# Set up Many-Body Tensor Representation object
mbtr = MBTR(
    # Species list
    species=species,
    # 
    geometry={"function": "inverse_distance"},
    grid={"min": 0, "max": 1, "n": 10, "sigma": 0.1},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    periodic=True,
    normalization="l2",
) 

# Transform data into fingerprint vector
training = []
testing = []
# Transforming train data 
for i, atom_train in enumerate(train_data.atoms):
    if i % 1000 == 0:
        print("Converted training data fingerprints:")
        print(i)
    train_fingerprint = mbtr.create(atom_train)
    training.append(train_fingerprint)

# Transforming test data
for i, atom_test in enumerate(test_data.atoms):
    if i % 1000 == 0:
        print("Converted test data fingerprints:")
        print(i)
    test_fingerprint = mbtr.create(atom_test)
    testing.append(test_fingerprint)

# Store in dataframe 
print("Store in dataframe")
X_train_i = pd.DataFrame(data=training, index = train_data.index)
X_test_i = pd.DataFrame(data=testing, index = test_data.index)
y = train_data.hform

print('X_train: {}'.format(X_train_i.shape))
print('X_test: {}'.format(X_test_i.shape))
print('y: {}'.format(y.shape))


# Reduce fingerprint dimensionality
from sklearn.decomposition import PCA 
print("PCA transform")

#### Control the number of PCA components
n_comp_PCA = 1500
####

pca = PCA(n_components = n_comp_PCA).fit(X_train_i)
X_train = pca.transform(X_train_i)
X_test = pca.transform(X_test_i)

print("With {} PCA components {var:0.4f}% of the variance is explained".format(n_comp_PCA, var = 100*np.sum(pca.explained_variance_ratio_)))
print('X_train: {}'.format(X_train.shape))
print('X_test: {}'.format(X_test.shape))

# Store training data and target data in dictionary
print("Store in dictionary")
data_dict = {}
data_dict["xp"] = X_train
data_dict["yp"] = y
data_dict["test"] = X_test 

# Save the data dictionary as pickle 
with open("data_dict.pkl", "wb") as f:
    pickle.dump(data_dict, f)


# Unpickling the dictionary
# with open('data.pkl', 'rb') as f:
#     data_loaded = pickle.load(f)
# print(data_loaded)




