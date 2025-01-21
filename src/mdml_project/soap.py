# %% [markdown]
# # Load the test and training data

# %%
import numpy as np
import pandas as pd
import json
from dscribe.descriptors import CoulombMatrix
from ase import Atoms
import matplotlib.pyplot as plt

test = pd.read_json("C:/1uni/ML for material design/Kaggle_competition/MDML_Project/data/processed/test.json")
train = pd.read_json("C:/1uni/ML for material design/Kaggle_competition/MDML_Project/data/processed/train.json")
#test = pd.read_json('data/test.json')
#train = pd.read_json('data/train.json')
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

# %%
max_number_of_atoms

# %%
from ase.visualize import view
view(train.atoms[1])

# %% [markdown]
# # Creating SOAP fingerprint

# %%
from dscribe.descriptors import SOAP,CoulombMatrix

r_cut = 6
n_max = 2
l_max = 2

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=True,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average = 'inner'
)

# %%
v = soap.create(train.atoms[1])

# %%
np.shape(v)

# %%
np.shape(np.nonzero(v))

# %%
soap_mats = np.zeros((len(train.atoms),23250))
for i,atoms in enumerate(train.atoms):
    if i%1000 == 1:
        print(i)
    soap_mats[i,:] = soap.create(atoms)
    

# %%
np.shape(soap_mats)

# %% [markdown]
# # Initialize X and y split into test and training

# %%
X = pd.DataFrame(data = soap_mats, index=train.id)
y = train['hform']
