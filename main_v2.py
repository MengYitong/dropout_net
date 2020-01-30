import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn import datasets
import data
import model
import pickle
import argparse
import os
from datetime import datetime
#epinions
# n_users = 1024
# n_items = 11601

#movielens-1m
n_users=6038
n_items=3395


def main():
    data_path = args.data_dir
    checkpoint_path = args.checkpoint_path
    tb_log_path = args.tb_log_path
    model_select = args.model_select
    num_epoch = args.num_epoch

    rank_out = args.rank
    user_batch_size = 100
    n_scores_user = 2500
    data_batch_size = args.batch_size#102
    dropout = args.dropout
    recall_at = [20,50]#range(50, 550, 50)
    eval_batch_size = 1000
    max_data_per_step = 2500000
    eval_every = args.eval_every
    # num_epoch = 20

    _lr = args.lr
    _decay_lr_every = 200
    _lr_decay = 0.5

    experiment = '%s_%s' % (
        datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        '-'.join(str(x / 100) for x in model_select) if model_select else 'simple'
    )
    _tf_ckpt_file = None if checkpoint_path is None else checkpoint_path + experiment + '/tf_checkpoint'

    print('running: ' + experiment)

    dat = load_data(data_path)
    u_pref_scaled = dat['u_pref_scaled']
    v_pref_scaled = dat['v_pref_scaled']
    # eval_warm = dat['eval_warm']
    # eval_cold_user = dat['eval_cold_user']
    eval_cold_item = dat['eval_cold_item']
    # user_content = dat['user_content']
    item_content = dat['item_content']
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    user_indices = dat['user_indices']

    timer = utils.timer(name='main').tic()

    # append pref factors for faster dropout
    v_pref_expanded = np.vstack([v_pref_scaled, np.zeros_like(v_pref_scaled[0, :])])
    v_pref_last = v_pref_scaled.shape[0]
    u_pref_expanded = np.vstack([u_pref_scaled, np.zeros_like(u_pref_scaled[0, :])])
    u_pref_last = u_pref_scaled.shape[0]
    timer.toc('initialized numpy data for tf')

    # prep eval
    eval_batch_size = eval_batch_size
    timer.tic()
    eval_cold_item.init_tf(u_pref_scaled, v_pref_scaled, None, item_content, eval_batch_size)
    timer.toc('initialized eval_cold_item for tf').tic()

    dropout_net = model.DeepCF(latent_rank_in=u_pref.shape[1],
                               user_content_rank=0,#user_content.shape[1],
                               item_content_rank=item_content.shape[1],
                               model_select=model_select,
                               rank_out=rank_out)

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device(args.model_device):
        dropout_net.build_model()

    with tf.device(args.inf_device):
        dropout_net.build_predictor(recall_at, n_scores_user)

    with tf.Session(config=config) as sess:
        tf_saver = None if _tf_ckpt_file is None else tf.train.Saver()
        train_writer = None if tb_log_path is None else tf.summary.FileWriter(
            tb_log_path + experiment, sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        row_index = np.copy(user_indices)
        n_step = 0
        # best_cold_user = 0
        # best_cold_item = 0
        # best_warm = 0
        n_batch_trained = 0
        best_step = 0
        for epoch in range(num_epoch):
            print('epoch:',epoch)
            np.random.shuffle(row_index)
            for b in utils.batch(row_index, user_batch_size):
                n_step += 1
                # print()
                # prep targets
                target_users = np.repeat(b, n_scores_user)
                target_users_rand = np.repeat(np.arange(len(b)), n_scores_user)
                target_items_rand = [np.random.choice(v_pref.shape[0], n_scores_user) for _ in b]
                target_items_rand = np.array(target_items_rand).flatten()
                target_ui_rand = np.transpose(np.vstack([target_users_rand, target_items_rand]))
                [target_scores, target_items, random_scores] = sess.run(
                    [dropout_net.tf_topk_vals, dropout_net.tf_topk_inds, dropout_net.preds_random],
                    feed_dict={
                        dropout_net.U_pref_tf: u_pref[b, :],
                        dropout_net.V_pref_tf: v_pref,
                        dropout_net.rand_target_ui: target_ui_rand
                    }
                )
                # merge topN and randomN items per user
                target_scores = np.append(target_scores, random_scores)
                target_items = np.append(target_items, target_items_rand)
                target_users = np.append(target_users, target_users)

                tf.local_variables_initializer().run()
                n_targets = len(target_scores)
                perm = np.random.permutation(n_targets)
                n_targets = min(n_targets, max_data_per_step)
                data_batch = [(n, min(n + data_batch_size, n_targets)) for n in xrange(0, n_targets, data_batch_size)]
                f_batch = 0
                for (start, stop) in data_batch:
                    batch_perm = perm[start:stop]
                    batch_users = target_users[batch_perm]
                    batch_items = target_items[batch_perm]
                    if dropout != 0:
                        n_to_drop = int(np.floor(dropout * len(batch_perm)))
                        perm_user = np.random.permutation(len(batch_perm))[:n_to_drop]
                        perm_item = np.random.permutation(len(batch_perm))[:n_to_drop]
                        batch_v_pref = np.copy(batch_items)
                        batch_u_pref = np.copy(batch_users)
                        batch_v_pref[perm_user] = v_pref_last # set drop_out to be 0
                        batch_u_pref[perm_item] = u_pref_last
                    else:
                        batch_v_pref = batch_items
                        batch_u_pref = batch_users

                    # print('u_pref_expanded[batch_u_pref, :]:',u_pref_expanded[batch_u_pref, :])
                    # print('v_pref_expanded[batch_v_pref, :]:',v_pref_expanded[batch_v_pref, :])
                    # print('item_content[batch_items, :]',item_content[batch_items, :])

                    # print('u_pref_expanded[batch_u_pref, :] is nan:',np.isnan(np.sum(u_pref_expanded[batch_u_pref, :])))
                    # print('v_pref_expanded[batch_v_pref, :] is nan:',np.isnan(np.sum(v_pref_expanded[batch_v_pref, :])))
                    # print('item_content[batch_items, :] is nan:',np.isnan(np.sum(item_content[batch_items, :])))
                    # print('target_scores[batch_perm]:',target_scores[batch_perm])

                    preds, _, loss_out,u_w,u_b,u_emb_w,u_emb_b,v_w,v_b,v_emb_w,v_emb_b,v_last= sess.run(
                        [dropout_net.preds, dropout_net.updates, dropout_net.loss,dropout_net.u_w,dropout_net.u_b,dropout_net.u_emb_w, dropout_net.u_emb_b,dropout_net.v_w,dropout_net.v_b,dropout_net.v_emb_w,dropout_net.v_emb_b,dropout_net.v_last],
                        feed_dict={
                            dropout_net.Uin: u_pref_expanded[batch_u_pref, :],
                            dropout_net.Vin: v_pref_expanded[batch_v_pref, :],
                            dropout_net.Vcontent: item_content[batch_items, :].todense(),
                            #
                            dropout_net.target: target_scores[batch_perm],
                            dropout_net.lr_placeholder: _lr,
                            dropout_net.phase: 1
                        }
                    )
                    # print('v_last:',v_last)
                    # print('u_w[0]:',u_w[0])
                    # print('u_b[0]:', u_b[0])
                    # print('u_w[1]:', u_w[1])
                    # print('u_b[1]:', u_b[1])
                    # print('u_emb_w:', u_emb_w.shape)
                    # print('u_emb_b:', u_emb_b.shape)
                    # print('v_emb_w:', v_emb_w.shape)
                    # print('v_emb_b:', v_emb_b.shape)
                    # print('V_embedding:', V_embedding.shape)
                    # print('preds:',preds)
                    if np.isnan(loss_out):
                        print('loss_out:',loss_out)
                    f_batch += loss_out
                    if np.isnan(f_batch):
                        raise Exception('f is nan')

                n_batch_trained += len(data_batch)
                if n_step % _decay_lr_every == 0: # learning rate decay
                    _lr = _lr_decay * _lr
                    print('decayed lr:' + str(_lr))
                if n_step % eval_every == 0:
                    recall_cold_item = utils.batch_eval_recall(
                        sess, dropout_net.eval_preds_cold,
                        eval_feed_dict=dropout_net.get_eval_dict,
                        recall_k=recall_at, eval_data=eval_cold_item)

                    # checkpoint
                    # if np.sum(recall_warm + recall_cold_user + recall_cold_item) > np.sum(
                    #                         best_warm + best_cold_user + best_cold_item):
                    #     best_cold_user = recall_cold_user
                    #     best_cold_item = recall_cold_item
                    #     best_warm = recall_warm
                    #     best_step = n_step
                    #     if tf_saver is not None:
                    #         tf_saver.save(sess, _tf_ckpt_file)

                    timer.toc('%d [%d]b [%d]tot f=%.2f best[%d]' % (
                        n_step, len(data_batch), n_batch_trained, f_batch, best_step
                    )).tic()
                    print ('\t\t\t'+' '.join([('@'+str(i)).ljust(6) for i in recall_at]))
                    # print('warm start\t%s\ncold user\t%s\ncold item\t%s' % (
                    #     ' '.join(['%.4f' % i for i in recall_warm]),
                    #     ' '.join(['%.4f' % i for i in recall_cold_user]),
                    #     ' '.join(['%.4f' % i for i in recall_cold_item])
                    # ))
                    print('cold item\t%s' % (
                        ' '.join(['%.4f' % i for i in recall_cold_item])
                    ))
                    summaries = []
                    for i, k in enumerate(recall_at):
                        if k % 100 == 0:
                            summaries.extend([
                                tf.Summary.Value(tag="recall@" + str(k) + " cold_item",
                                                 simple_value=recall_cold_item[i])
                            ])
                    recall_summary = tf.Summary(value=summaries)
                    if train_writer is not None:
                        train_writer.add_summary(recall_summary, n_step)

        now = datetime.now()

        # print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        # print("date and time =", dt_string)
        with open('./result/results'+dt_string+'.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([u_w,u_b,u_emb_w,u_emb_b,v_w,v_b,v_emb_w,v_emb_b], f)
            print('result saved.')


def load_data(data_path):
    timer = utils.timer(name='main').tic()
    split_folder = os.path.join(data_path, 'warm')

    u_file = os.path.join(data_path, 'trained/warm/U.csv.bin')
    v_file = os.path.join(data_path, 'trained/warm/V.csv.bin')
    # user_content_file = os.path.join(data_path, 'user_features_0based.txt')
    item_content_file = os.path.join(data_path, 'item_features_0based.txt')
    train_file = os.path.join(split_folder, 'train.csv')
    # test_warm_file = os.path.join(split_folder, 'test_warm.csv')
    # test_warm_iid_file = os.path.join(split_folder, 'test_warm_item_ids.csv')
    # test_cold_user_file = os.path.join(split_folder, 'test_cold_user.csv')
    # test_cold_user_iid_file = os.path.join(split_folder, 'test_cold_user_item_ids.csv')
    test_cold_item_file = os.path.join(split_folder, 'test_cold_item.csv')
    test_cold_item_iid_file = os.path.join(split_folder, 'test_cold_item_item_ids.csv')

    dat = {}
    # load preference data
    timer.tic()
    u_pref = np.fromfile(u_file, dtype=np.float32).reshape(n_users, 200)
    v_pref = np.fromfile(v_file, dtype=np.float32).reshape(n_items, 200)
    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    timer.toc('loaded U:%s,V:%s' % (str(u_pref.shape), str(v_pref.shape))).tic()

    # pre-process
    _, dat['u_pref_scaled'] = utils.prep_standardize(u_pref)
    _, dat['v_pref_scaled'] = utils.prep_standardize(v_pref)
    timer.toc('standardized U,V').tic()

    # load content data
    # timer.tic()
    # user_content, _ = datasets.load_svmlight_file(user_content_file, zero_based=True, dtype=np.float32)
    # dat['user_content'] = user_content.tolil(copy=False)
    # timer.toc('loaded user feature sparse matrix: %s' % (str(user_content.shape))).tic()
    item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
    dat['item_content'] = item_content.tolil(copy=False)
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()

    # load split
    timer.tic()
    # train = pd.read_csv(train_file, delimiter=",", header=1, dtype=np.int32).values.ravel().view(
    #     dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32), ('date', np.int32)])
    train = pd.read_csv(train_file, delimiter=",", header=0,usecols=[0, 1, 2] , dtype=np.int32).values.ravel().view(
        dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])
    # print('train:',train)
    # exit()
    dat['user_indices'] = np.unique(train['uid'])
    timer.toc('read train triplets %s' % train.shape).tic()

    # dat['eval_warm'] = data.load_eval_data(test_warm_file, test_warm_iid_file, name='eval_warm', cold=False,
    #                                        train_data=train)
    # dat['eval_cold_user'] = data.load_eval_data(test_cold_user_file, test_cold_user_iid_file, name='eval_cold_user',
    #                                             cold=True,
    #                                             train_data=train)
    dat['eval_cold_item'] = data.load_eval_data(test_cold_item_file, test_cold_item_iid_file, name='eval_cold_item',
                                                cold=True,
                                                train_data=train,citeu=True)
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script to run DropoutNet on RecSys data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='path to eval in the downloaded folder')

    parser.add_argument('--model-device', type=str, default='/gpu:0', help='device to use for training')
    parser.add_argument('--inf-device', type=str, default='/cpu:0', help='device to use for inference')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='path to dump checkpoint data from TensorFlow')
    parser.add_argument('--tb-log-path', type=str, default=None,
                        help='path to dump TensorBoard logs')
    parser.add_argument('--model-select', nargs='+', type=int,
                        default=[10,5],#800,400
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units',
                        )
    parser.add_argument('--rank', type=int, default=500, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='DropoutNet dropout')
    parser.add_argument('--eval-every', type=int, default=10, help='evaluate every X user-batch')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--num_epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='number of training epochs')

    args = parser.parse_args()
    main()
