import numpy as np

batch = 5
number = 3
data = []
global_data1 = []
global_data2 = []
local_data = []
for i in range(batch):
    per_local_data = []

    a = np.zeros((1,2,4,8))
    print(a[None].shape)
    data.append(a[None])
    global_data1.append(a[None])
    global_data2.append(a[None])
    for j in range(number):
        per_local_data.append(a[None])
    per_local_data= np.vstack(per_local_data)
    print(per_local_data.shape)
    local_data.append(per_local_data[None])


data = np.vstack(data)
global_data1 = np.vstack(global_data1)
global_data2 = np.vstack(global_data2)
local_data = np.vstack(local_data)
print(data.shape)
print(global_data1.shape)
print(global_data2.shape)
print(local_data.shape)


