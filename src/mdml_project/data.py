import json
import pandas as pd
import numpy as np
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def load_data():
    # load a json file
    train = pd.read_json('data/processed/train.json')
    test = pd.read_json('data/processed/test.json')

    train.atoms = train.atoms.apply(lambda x: Atoms(**x)) # OBS This one is important!
    test.atoms = test.atoms.apply(lambda x: Atoms(**x))

    species = []
    number_of_atoms = []
    atomic_numbers = []
    for atom in pd.concat([train.atoms,test.atoms]):
        species = list(set(species+atom.get_chemical_symbols()))
        atomic_numbers = list(set(atomic_numbers+list(atom.get_atomic_numbers())))
        number_of_atoms.append(len(atom))

    max_number_of_atoms = np.max(number_of_atoms)
    min_atomic_number = np.min(atomic_numbers)
    max_atomic_number = np.max(atomic_numbers)

    print(len(species))

    # Setting up the CM descriptor
    cm = CoulombMatrix(
        n_atoms_max=max_number_of_atoms,
    )

    cmats = np.zeros((len(train),max_number_of_atoms**2))
    for i,atoms in enumerate(train.atoms):
        if i%1000 == 0:
            print(i)
        cmats[i,:] = cm.create(atoms)
    print(len(cmats))

    X = pd.DataFrame(data = cmats, index=train.id)
    y = train['hform']
    print('X: {}'.format(X.shape))
    print('y: {}'.format(y.shape))

    # Standardize the data
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)

    print('X: {}'.format(X.shape)) 
    print('y: {}'.format(y.shape))

    #y = y.values

    print('Data loaded successfully')
    return X, y