# %% [markdown]
# ### Imports 

# %%
import numpy as np
import pandas as pd
import json
import dscribe
from dscribe.descriptors import CoulombMatrix
from ase import Atoms
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 22}

matplotlib.rc('font', **font)



# %% [markdown]
# ## Loading data and setting up the data frames

# %%
#data_dir = "data/" # Specify your data path (Folder in which the files are placed)

# %%
# Loading the data as pandas DataFrame
test = pd.read_json("C:/1uni/ML for material design/Kaggle_competition/MDML_Project/data/processed/test.json")
train = pd.read_json("C:/1uni/ML for material design/Kaggle_competition/MDML_Project/data/processed/train.json")
## Transform atoms entry to ASE atoms object
train.atoms = train.atoms.apply(lambda x: Atoms(**x)) # OBS This one is important!
test.atoms = test.atoms.apply(lambda x: Atoms(**x))

# %%
print('Train data shape: {}'.format(train.shape))
train.head()

# %%
print('Test data shape: {}'.format(test.shape))
test.head()

# %%
train.describe()

# %% [markdown]
# ## Creating the Coulomb matrix fingerprint
# #### First a preprocessing step

# %%
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

print(max_number_of_atoms)

# %% [markdown]
# #### Coulomb matrix
# DScribe: Python package for transforming ASE Atoms into fingerprints!
# https://singroup.github.io/dscribe/latest/
# 
# Note: This package is built for Linux/Mac and can be difficult to install on Windows and M1 Macs. See the instructions at the end of this notebook.

# %%
# Setting up the CM descriptor
cm = CoulombMatrix(
    n_atoms_max=max_number_of_atoms,
)

# %%
cmats = np.zeros((len(train),max_number_of_atoms**2))
for i,atoms in enumerate(train.atoms):
    if i%1000 == 0:
        print(i)
    cmats[i,:] = cm.create(atoms)
print(len(cmats))

# %%
cmats.shape

# %% [markdown]
# # Setting target and feature vector

# %%
X = pd.DataFrame(data = cmats, index=train.id)
y = train['hform']
print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))

# %% [markdown]
# ## Splitting into test and train set

print("Coulomb ran successfully")

