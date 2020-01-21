import pickle,numpy as np,os
from sklearn import datasets
data_dir='./data/DropoutNet_epin/'
with open('./result/results.pkl','rb') as f:
    u_w,u_b,u_emb_w,u_emb_b,v_w,v_b,v_emb_w,v_emb_b = pickle.load(f)
test_id_file='warm/test_cold_item_item_ids.csv'
u_file = os.path.join(data_dir, 'trained/warm/U.csv.bin')
v_file = os.path.join(data_dir, 'trained/warm/V.csv.bin')
v_content_file=os.path.join(data_dir, 'item_features_0based.txt')
V_content, _ = datasets.load_svmlight_file(v_content_file, zero_based=True, dtype=np.float32)
U=np.fromfile(u_file, dtype=np.float32).reshape(-1, 200)
V=np.fromfile(v_file, dtype=np.float32).reshape(-1, 200)
# print(u_w[0].shape)
# print(u_w[1].shape)
# print(u_emb_w.shape)
U_last=U
for i in range(len(u_w)):
    U_last = np.dot(U_last, u_w[i] )+ u_b[i]
U_out=np.dot(U_last ,u_emb_w )+ u_emb_b
# print(U_out.shape)
# print(U_out)
print('V.shape:',V.shape)
print('V_content.shape:',V_content.shape)
V_concat = np.concatenate([V, V_content], axis=1)
print(V_concat.shape)