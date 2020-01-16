import numpy as np
import os
n_items = 1306054 + 1
data_path='./data/recsys/'
file='trained/warm/V.csv.bin'
test_id_file='warm/test_cold_item_item_ids.csv'
u_file = os.path.join(data_path, file)
U = np.fromfile(u_file, dtype=np.float32).reshape(-1, 200)

with open(test_id_file) as f:
    test_item_ids = [int(line) for line in f]

U=abs(U)
U=U[test_item_ids,:]
print(U.shape)
U=np.sum(U,axis=1)
nnz=np.count_nonzero(U)
print('file:',file)
print(U.shape)
print('nnz',nnz)
# print(u_pref[:10000])
# print(u_pref[8,:])
