import numpy as np
import pandas as pd 
import pickle 

### Name of datafile to be read 
data_file = "datadata_dict.pkl"

# Global parameters 
l = 1.4
k0 = 1
sigma = 0.003

cwd = __file__.split("/")[1:-1]
data_dir = ""
for dir in cwd:
    data_dir += "/" + dir 
    if dir == "MDML_Project":
        break 

data_dir +="/" + "data" + "/" + data_file
print("")
print(data_dir)
print("")

# Import data
print("Loading in data")
with open(data_dir, "rb") as f:
    data_dict = pickle.load(f)

xp = data_dict["xp"]
tp = data_dict["yp"]



# Build kernel 
print("Building kernel")
def kernel(x, x0, l, k0):
    eks = np.dot((x-x0), (x-x0))
    return k0*np.exp(-(eks) / (2*l**2))

print("Defining Kvec")
def Kvec(x):
    k_vec = np.array([kernel(x, x0, l, k0) for x0 in xp])
    return k_vec 


# Build C 
print("Buildiing C-matrix")
Kmat = np.zeros((len(xp), len(xp)))

for i, vec1 in enumerate(xp):
    for j, vec2 in enumerate(xp):
        Kmat[i,j] = kernel(vec1, vec2, l, k0)

C = Kmat + sigma**2 * np.identity(len(xp))
print("Inverting C matrix")
Cinv = np.linalg.inv(C)

print("Cinv built")
print("Proceeding to prediction")

def y0(x):
    kx = np.transpose(Kvec(x))
    Ct = Cinv @ tp 
    return kx @ Ct 


test_vec = data_dict["test"]
ids = data_dict["test_ids"]

predicted = []
id_list = []
print("Starting prediction")
for i, (id, vector) in enumerate(zip(ids, test_vec)):
    if i % 1000 == 0:
        print(i)

    id_list.append(id)
    predicted.append(y0(vector))

print("Finished prediction")
print("Putting data into dataframe")

data = {"id": id_list, "hform": predicted}
df = pd.DataFrame(data)
print("Writing file")
df.to_csv("GS_prediction_MBTR.csv", index=False)
print("All finished")
