import pickle,numpy as np,os
from sklearn import datasets
from scipy.sparse import  csr_matrix
import evaluate as ev
import utils

def reconstruct_ratings(u_file,v_file,v_content_file,test_id_file,result_path):
    with open(result_path,'rb') as f:
        u_w,u_b,u_emb_w,u_emb_b,v_w,v_b,v_emb_w,v_emb_b = pickle.load(f)

    V_content, _ = datasets.load_svmlight_file(v_content_file, zero_based=True, dtype=np.float32)
    V_content=V_content.A*9
    U=np.fromfile(u_file, dtype=np.float32).reshape(-1, 200)
    V=np.fromfile(v_file, dtype=np.float32).reshape(-1, 200)
    print('U:', U)
    print('V:', V)
    _,U=utils.prep_standardize(U)
    # _,V=utils.prep_standardize(V)

    print('U:',U)
    print('V:', V)
    # exit()
    # print(u_w[0])
    # print(u_w[1].shape)
    # print(u_emb_w.shape)
    U_last=U
    for i in range(len(u_w)):
        U_last = np.dot(U_last, u_w[i] )+ u_b[i]
        U_last = np.tanh(U_last)
        print('U_last:', U_last)
    U_out=np.dot(U_last ,u_emb_w )+ u_emb_b
    # print(U_out.shape)
    # print(U_out)
    # print('V.shape:',V.ndim)
    # print('V_content.shape:',V_content.ndim)

    with open(test_id_file) as f:
        test_item_ids = [int(line) for line in f]

    V_concat = np.concatenate((V, V_content), axis=1)
    # print(V_concat)
    V_last=V_concat
    # print('V_concat:')
    # print(V_concat[test_item_ids, :])
    # print('v_w[0]:\n',v_w[0])
    for i in range(len(v_w)):
        # print(i)
        V_last = np.dot(V_last, v_w[i] )+ v_b[i]
        V_last=np.tanh(V_last)
        # print('v_b[i]:',v_b[i])
        print('V_last:',V_last)
    V_out=np.dot(V_last ,v_emb_w )+ v_emb_b

    # print('V_out[test_item_ids,:]:',V_out[test_item_ids,:].T)
    # exit()
    # print('V_out.shape:',V_out.ndim)
    pred=np.dot(U_out,V_out.T)
    return pred

def eval_dropoutnet(whole_pred,test_id_file,test_rating_path):
    with open(test_id_file) as f:
        test_item_ids = [int(line) for line in f]

    x=np.loadtxt(test_rating_path,delimiter=',')
    row = x[:, 0].astype(int)
    col = x[:, 1].astype(int)
    data = x[:, 2]
    ratings = csr_matrix((data, (row, col)), dtype=np.float32)
    test=ratings[:,test_item_ids]
    test=test.A
    pred=(whole_pred[:,test_item_ids])#+0.0000001*test
    print('pred:',pred)
    map, ndcg, recall = ev.eval2(pred, test)
    print('map:', map, ', ndcg:', ndcg, ', recall:', recall)

if __name__ == "__main__":
    # result_path='./result/results.pkl'
    result_path='./result/results20200118_163236.pkl'

    data_dir = './data/DropoutNet_epin/'
    test_id_file=os.path.join(data_dir,'warm/test_cold_item_item_ids.csv')
    test_rating_path=os.path.join(data_dir,'warm/test_cold_item.csv')
    u_file = os.path.join(data_dir, 'trained/warm/U.csv.bin')
    v_file = os.path.join(data_dir, 'trained/warm/V.csv.bin')
    v_content_file=os.path.join(data_dir, 'item_features_0based.txt')

    whole_pred=reconstruct_ratings(u_file, v_file, v_content_file,test_id_file, result_path)
    eval_dropoutnet(whole_pred, test_id_file, test_rating_path)
