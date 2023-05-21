import tensorflow as tf
import numpy as np
import h5py, math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import *
from nn_model import deep_nn_tmp
from sklearn.preprocessing import StandardScaler
import pickle, os


def HBF_Eva(ns_s, ns_t, data, bf, ind_test, y_test, snr_out, flag):
    """
    Evaluating HBS performance
    Inputs:
            ns_s:
            ns_t:
            data:
            bf:
            ind_test:
            y_test:
            snr_out:
        flag: 'd'=>DEACT,  'b'=>BMW-SS, 'd_r'=>DEACT reverse,  'b_r'=>BMW-SS reverse
    """
    print('Start calculating accuracy ...')
    data_test = data[ind_test, :]
    ind_sort_2 = data_test[:, bf['ind_s']]
    rss_2 = data_test[:, bf['rss']]
    snr_2 = data_test[:, bf['snr']]

    align_0db = 0
    align_3db = 0
    outage_perfect_num = 0
    outage_num = 0
    ave_achievable_rate = 0
    ave_perfect_align_rate = 0
    ave_perfect_snr = 0

    for m in range(np.shape(ind_test)[0]):
        ind_s = (ind_sort_2[m, :]).astype(int)
        if ns_s != ns_t:
            d = data_test[m, bf['ave_rss']] * np.ones([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = data_test[m, bf['ave_snr']] * np.ones([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        else:
            d = np.zeros([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = np.zeros([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        snr_max = np.amax(snr)
        ave_perfect_align_rate += math.log2(1 + snr_max)
        ave_perfect_snr += snr_max
        if snr_max < snr_out:
            outage_perfect_num += 1
        """ 0dB and 3dB power loss probability """
        rss_top_n = d[int(data_test[m, bf['idx_'+flag]])]
        if rss_top_n >= d[y_test[m]]:
            align_0db += 1
        if rss_top_n > 0.5 * d[y_test[m]]:
            align_3db += 1
        snr_ = snr[int(data_test[m, bf['idx_'+flag]])]
        ave_achievable_rate += math.log2(1 + snr_)
        if snr_ < snr_out:
            outage_num += 1

    num_iter = float(np.shape(ind_test)[0])
    prob_align_0db = align_0db / num_iter
    prob_align_3db = align_3db / num_iter
    prob_ave_achievable_rate = float(ave_achievable_rate) / num_iter
    prob_out = float(outage_num) / (num_iter + 1e-10)
    prob_out_perfect = float(outage_perfect_num) / (num_iter + 1e-10)
    prob_ave_perfect_align_rate = float(ave_perfect_align_rate) / num_iter

    return prob_align_0db, prob_align_3db, prob_ave_perfect_align_rate, prob_ave_achievable_rate, \
        prob_out_perfect, prob_out


if __name__ == '__main__':
    src_nr = 60  # source dataset
    dst_nr = 61  # destination dataset

    src_file_nr = src_nr
    dst_file_nr = dst_nr
    src_file_p = '3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.p'.format(src_file_nr)
    src_file_d = '3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.hdf5'.format(src_file_nr)
    dst_file_p = '3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.p'.format(dst_file_nr)
    dst_file_d = '3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.hdf5'.format(dst_file_nr)
    print('src: ', src_file_d)
    print('dst: ', dst_file_d)
    """
        Source dataset - Loading settings
    """
    info_s = pickle.load(open(src_file_p, "rb"))[0]
    params_s = info_s['params']
    bf_s = info_s['beamform_info']

    """
        Destination dataset - Loading settings
    """
    info_d = pickle.load(open(dst_file_p, "rb")) [0]
    params_d = info_d['params']
    bf_d = info_d['beamform_info']
    len_HBS = 2*(np.sum(np.log2(params_d['nr'])) + np.sum(np.log2(params_d['nt'])))
    """ 
        Parameters of the neural networks
    """
    ns_s = params_d['ns_s']  # Information of ns_s best beam pairs is available in the dataset
    nb_t = params_d['nb_t']  # Number of beams at TX
    nb_r = params_d['nb_r']  # Number of beams at RX
    ns_t = prod(nb_t) * prod(nb_r)  # Number of beam pairs
    ns_t_load = prod(params_s['nb_t']) * prod(params_s['nb_r'])  # Number of beam pairs in the source setting
    multilabel = np.arange(1, 2, 1)  # Labeling mode (default = 1)
    """ 
        Parameters of the metrics
    """
    snr_out_dB = 10  # Threshold snr for outage (dB)
    snr_out = 10 ** (snr_out_dB / 10)
    top_n = np.concatenate((np.arange(1, 10, 1), np.arange(10, 20, 5), np.arange(20, 51, 10)), axis=0)  # Number of beam pairs scanned (N_b)
    """
        Source dataset
    """
    BS_loc_s = np.array(params_s['loc_tx'])  # Location of the BS at the source setting
    flag_source = 0  # Controlling whether the source network should be trained or not
    multilabel_s = 1  # Labeling mode for the source setting (default = 1)
    print('Loading source data ...')
    with h5py.File(src_file_d,'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data = np.zeros([s[1], s[2]], dtype=float)
        data[:, ] = data_r[0, :, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    data_s = data
    x_s = np.zeros([np.shape(data_s)[0], 5])
    x_s[:, 0] = np.arange(np.shape(data_s)[0])
    x_s[:, 1:4] = data_s[:, bf_s['loc_rx']] - BS_loc_s  # Position of RXs
    x_s[:, 4] = data_s[:, bf_s['rot_rx'][0]]  # Orientation of RXs
    y_s = data_s[:, bf_s['ind_s'][0]]  # Labels for each RX point
    y_s = y_s.astype(int)

    x_train_s, x_val_test_s, y_train_s, y_val_test_s = train_test_split(x_s, y_s, test_size=0.2, random_state=42)
    x_val_s, x_test_s, y_val_s, y_test_s = train_test_split(x_val_test_s, y_val_test_s, test_size=0.5, random_state=42)

    ind_train_s = x_train_s[:, 0]
    ind_train_s = ind_train_s.astype(int)
    ind_val_test_s = x_val_test_s[:, 0]
    ind_val_test_s = ind_val_test_s.astype(int)
    ind_val_s = x_val_s[:, 0]
    ind_val_s = ind_val_s.astype(int)
    ind_test_s = x_test_s[:, 0]
    ind_test_s = ind_test_s.astype(int)

    x_train_s = x_train_s[:, 1:]
    x_val_s = x_val_s[:, 1:]
    x_val_test_s = x_val_test_s[:, 1:]

    """
        Destination dataset
    """
    BS_loc_d = np.array(params_d['loc_tx'])
    if all(np.array(params_s['nt']) == np.array(params_d['nt'])) and all(np.array(params_s['nr']) == np.array(params_d['nr'])):
        f_ft = 0  # 0: copy the weights of the output layer from the source network, 1: random initialization for the output layer
        Fr_list = [(0, 0), (0, 1)]  # list of freezing settings. (a, b) means freezing parameters in the a first and b last layers
    else:
        f_ft = 1  # 0: copy the weights of the output layer from the source network, 1: random initialization for the output layer
        Fr_list = [(0, 0), (5, 0)]   # list of freezing settings. (a, b) means freezing parameters in the a first and b last layers
    with h5py.File(dst_file_d,'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data = np.zeros([s[1], s[2]], dtype=float)
        data[:, ] = data_r[0, :, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    data_d = data
    x_d = np.zeros([np.shape(data_d)[0], 5])
    x_d[:, 0] = np.arange(np.shape(data_d)[0])
    x_d[:, 1:4] = data_d[:, bf_d['loc_rx']] - BS_loc_d  # Position of RXs
    x_d[:, 4] = data_d[:, bf_d['rot_rx'][0]]  # Orientation of RXs
    y_d = data_d[:, bf_d['ind_s'][0]]  # Labels for each RX point
    y_d = y_d.astype(int)

    x_train_d, x_val_test_d, y_train_d, y_val_test_d = train_test_split(x_d, y_d, test_size=0.2, random_state=42)
    x_val_d, x_test_d, y_val_d, y_test_d = train_test_split(x_val_test_d, y_val_test_d, test_size=0.5, random_state=42)

    ind_train_d = x_train_d[:, 0]
    ind_train_d = ind_train_d.astype(int)
    ind_val_test_d = x_val_test_d[:, 0]
    ind_val_test_d = ind_val_test_d.astype(int)
    ind_val_d = x_val_d[:, 0]
    ind_val_d = ind_val_d.astype(int)
    ind_test_d = x_test_d[:, 0]
    ind_test_d = ind_test_d.astype(int)

    x_train_d = x_train_d[:, 1:]
    x_val_d = x_val_d[:, 1:]
    x_val_test_d = x_val_test_d[:, 1:]


    x_train_d_prepared = x_train_d
    x_val_d_prepared = x_val_d
    x_val_test_d_prepared = x_val_test_d

    ds = np.shape(x_train_d)[0]/100
    if src_nr == 42 and dst_nr == 51:
        dataset_sizes = [0, int(1 * ds), int(5 * ds), int(10 * ds), -1]  # Dataset sizes for evaluations
    else:
        dataset_sizes = [0, int(5*ds), int(10*ds), int(20*ds), -1]  # Dataset sizes for evaluations

    """ 
        Other parameters
    """
    flag_clean = 0
    marker_list = ['o', 's', '*', 'd', 'v', '^', '<', '>', 'p', 'X', '8', 'h']
    color_list = [[0, 0.4470, 0.7410],
                  [0.8500, 0.3250, 0.0980],
                  [0.9290, 0.6940, 0.1250],
                  [0.4940, 0.1840, 0.5560],
                  [0.4660, 0.6740, 0.1880],
                  [0.3010, 0.7450, 0.9330],
                  [0.6350, 0.0780, 0.1840],
                  [0.2500, 0.2500, 0.2500]]

    nb = np.shape(top_n)[0]
    n_rows = 36 * np.shape(multilabel)[0]
    align_0db_t = np.zeros([n_rows, nb])
    align_3db_t = np.zeros([n_rows, nb])
    prob_out_perfect = np.zeros([n_rows, nb])
    prob_out = np.zeros([n_rows, nb])
    ave_perfect_align_rate_t = np.zeros([n_rows, nb])
    ave_achievable_rate_t = np.zeros([n_rows, nb])

    leg = []
    color_l = []
    marker_l = []
    c = 0
    counter = []

    """
        Training and evaluating of the source dataset
    """
    if flag_source == 0:
        if not os.path.isfile("3DRaytatracing_Models/model_3d_{:04d}_1hot_{}label.h5".format(src_nr, multilabel_s)):
            flag_source = 1

    if flag_source != 0:
        print('-----------------------100per - Source-----------------------')
        scaler_pre = StandardScaler()
        scaler_pre.fit(x_train_s)  # Calculating parameters for Normalization

        x_train_s_pre = scaler_pre.transform(x_train_s)  # Normalization
        x_val_s_pre = scaler_pre.transform(x_val_s)  # Normalization
        x_val_test_s_pre = scaler_pre.transform(x_val_test_s)  # Normalization

        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect[
            c], \
        prob_out[c] = deep_nn_tmp(ns_s, ns_t_load, top_n, data_s, ind_train_s, x_train_s_pre, y_train_s, bf_s,
                                  ind_val_s, x_val_s_pre, y_val_s, ind_val_test_s, x_val_test_s_pre, y_val_test_s,
                                  multilabel_s, 0, snr_out, flag_train=0,
                                  load_name="3DRaytatracing_Models/model_3d_{:04d}_1hot_{}label.h5".format(src_nr, multilabel_s))
        print('DNN-0db', align_0db_t[c])
        print('DNN-Ach', ave_achievable_rate_t[c])
        print('DNN-Pout', prob_out[c])


    """
        Training and evaluating of the destination dataset
    """
    for q in range(np.shape(dataset_sizes)[0]):
        if dataset_sizes[q] == -1:
            ds = np.shape(x_train_d)[0]
        elif dataset_sizes[q] == 0:
            ds = 1
        else:
            ds = dataset_sizes[q]

        x_train = x_train_d[:ds, :]
        ind_train = ind_train_d[:ds]

        scaler = StandardScaler()
        scaler.fit(x_train)  # Calculating parameters for Normalization
        x_train_n = scaler.transform(x_train)  # Normalization
        x_val_n = scaler.transform(x_val_d)  # Normalization
        x_val_test_n = scaler.transform(x_val_test_d)  # Normalization
        if dataset_sizes[q] != 0:
            for z in range(np.shape(multilabel)[0]):
                n_iter = 3
                align_0db_t_i = np.zeros([n_iter, nb])
                align_3db_t_i = np.zeros([n_iter, nb])
                prob_out_perfect_i = np.zeros([n_iter, nb])
                prob_out_i = np.zeros([n_iter, nb])
                ave_perfect_align_rate_t_i = np.zeros([n_iter, nb])
                ave_achievable_rate_t_i = np.zeros([n_iter, nb])
                for mm in range(n_iter):
                    print('----------------------- Destination - {} samples - iter {} -----------------------'.format(
                        dataset_sizes[q], mm))
                    h, align_0db_t_i[mm], align_3db_t_i[mm], ave_perfect_align_rate_t_i[mm], ave_achievable_rate_t_i[mm], \
                    prob_out_perfect_i[mm], prob_out_i[mm] = deep_nn_tmp(ns_s, ns_t, top_n, data_d, ind_train, x_train_n,
                                                                         y_train_d, bf_d, ind_val_d, x_val_n, y_val_d,
                                                                         ind_val_test_d, x_val_test_n, y_val_test_d,
                                                                         multilabel[z], 0, snr_out, flag_train=0,
                                                                         load_name="3DRaytatracing_Models/model_3d_{:04d}_{}s_1hot_{}label_{}.h5".format(dst_nr,
                                                                             ds, multilabel[z], mm))
                align_0db_t[c] = np.mean(align_0db_t_i, axis=0)
                align_3db_t[c] = np.mean(align_3db_t_i, axis=0)
                prob_out_perfect[c] = np.mean(prob_out_perfect_i, axis=0)
                prob_out[c] = np.mean(prob_out_i, axis=0)
                ave_perfect_align_rate_t[c] = np.mean(ave_perfect_align_rate_t_i, axis=0)
                ave_achievable_rate_t[c] = np.mean(ave_achievable_rate_t_i, axis=0)
                print('DNN-0db', align_0db_t[c])
                print('DNN-Ach', ave_achievable_rate_t[c])
                print('DNN-Pout', prob_out[c])
                c += 1
                if dataset_sizes[q] == -1:
                    leg.append('DNN-train-{}% of samples ({})'.format(int(np.round(100*np.shape(x_train_n)[0] / int(np.shape(x_train_d)[0]))), int(np.shape(x_train_d)[0])))
                else:
                    leg.append('DNN-train-{}% of samples'.format(int(np.round(100*np.shape(x_train_n)[0]/int(np.shape(x_train_d)[0])))))
                counter.append(1 + 3 * q + 10 * z)

        """
            Training neural networks using Transfer learning
        """
        if ds == 0:
            continue
        x_train_prepared = x_train_d_prepared[:ds, :]

        scaler_pre = StandardScaler()
        scaler_pre.fit(x_train_s)  # Calculating parameters for Normalization
        x_train_n_pre = scaler_pre.transform(x_train_prepared)  # Normalization
        x_val_n_pre = scaler_pre.transform(x_val_d_prepared)  # Normalization
        x_val_test_n_pre = scaler_pre.transform(x_val_test_d_prepared)  # Normalization

        for nn in range(len(Fr_list)):
            k = Fr_list[nn][0]
            k_l = Fr_list[nn][1]
            for z in range(np.shape(multilabel)[0]):
                n_iter = 3
                align_0db_t_i = np.zeros([n_iter, nb])
                align_3db_t_i = np.zeros([n_iter, nb])
                prob_out_perfect_i = np.zeros([n_iter, nb])
                prob_out_i = np.zeros([n_iter, nb])
                ave_perfect_align_rate_t_i = np.zeros([n_iter, nb])
                ave_achievable_rate_t_i = np.zeros([n_iter, nb])
                for mm in range(n_iter):
                    print('-------------------- Destination - TL - {} samples - iter {} ----------------------'.format(
                        dataset_sizes[q], mm))
                    if dataset_sizes[q] == 0 and z == 0:
                        h, align_0db_t_i[mm], align_3db_t_i[mm], ave_perfect_align_rate_t_i[mm], ave_achievable_rate_t_i[
                            mm], prob_out_perfect_i[mm], prob_out_i[mm] = deep_nn_tmp(ns_s, ns_t, top_n, data_d, ind_train, x_train_n_pre, y_train_d, bf_d, ind_val_d,
                                    x_val_n_pre, y_val_d, ind_val_test_d, x_val_test_n_pre, y_val_test_d, multilabel[z],
                                    0, snr_out, flag_freeze=k, flag_freeze_last=0, flag_ft=f_ft,
                                    ns_t_load=ns_t_load,
                                    load_name="3DRaytatracing_Models/model_3d_{:04d}_1hot_{}label.h5".format(src_nr, multilabel_s))
                    elif dataset_sizes[q] != 0:
                        if k == 0 and k_l == 0:
                            s_n = "3DRaytatracing_Models/model_TL_Fr_{}_3d_{:04d}_{:04d}_{}s_1hot_{}label_{}.h5".format(k, src_nr, dst_nr, ds, multilabel[z], mm)
                        else:
                            s_n = "3DRaytatracing_Models/model_TL_Fr_{}_{}_3d_{:04d}_{:04d}_{}s_1hot_{}label_{}.h5".format(k, k_l, src_nr, dst_nr, ds, multilabel[z], mm)
                        h, align_0db_t_i[mm], align_3db_t_i[mm], ave_perfect_align_rate_t_i[mm], ave_achievable_rate_t_i[
                            mm], prob_out_perfect_i[mm], prob_out_i[mm] = deep_nn_tmp(ns_s, ns_t, top_n, data_d, ind_train, x_train_n_pre, y_train_d, bf_d, ind_val_d,
                                        x_val_n_pre, y_val_d, ind_val_test_d, x_val_test_n_pre, y_val_test_d, multilabel[z],
                                        0, snr_out, flag_freeze=k, flag_freeze_last=k_l, flag_train=0, flag_ft=f_ft,
                                        ns_t_load=ns_t_load, load_name="3DRaytatracing_Models/model_3d_{:04d}_1hot_{}label.h5".format(src_nr, multilabel_s),
                                        save_name=s_n)
                align_0db_t[c] = np.mean(align_0db_t_i, axis=0)
                align_3db_t[c] = np.mean(align_3db_t_i, axis=0)
                prob_out_perfect[c] = np.mean(prob_out_perfect_i, axis=0)
                prob_out[c] = np.mean(prob_out_i, axis=0)
                ave_perfect_align_rate_t[c] = np.mean(ave_perfect_align_rate_t_i, axis=0)
                ave_achievable_rate_t[c] = np.mean(ave_achievable_rate_t_i, axis=0)
                print('DNN-TL-0db', align_0db_t[c])
                print('DNN-TL-Ach', ave_achievable_rate_t[c])
                print('DNN-TL-Pout', prob_out[c])
                c += 1
                if k == 0:
                    if dataset_sizes[q] == 0:
                        leg.append('DNN-TL-No fine tuning'.format(np.shape(x_train_n_pre)[0]))
                    elif dataset_sizes[q] == -1:
                        leg.append('DNN-TL-fine tuning-{}% of samples ({})'.format(int(np.round(100*np.shape(x_train_n_pre)[0] / int(np.shape(x_train_d)[0]))), int(np.shape(x_train_d)[0])))
                    else:
                        leg.append('DNN-TL-fine tuning-{}% of samples'.format(int(np.round(100*np.shape(x_train_n_pre)[0]/int(np.shape(x_train_d)[0])))))
                elif k == 4:
                    leg.append('DNN-TL-fine tuning-{} samples - Freeze Hidden Layers'.format(np.shape(x_train_n_pre)[0]))
                else:
                    leg.append('DNN-TL-fine tuning-{} samples - Freeze {} First Layer(s)'.format(np.shape(x_train_n_pre)[0], k))
                counter.append(2 + int(k / 4) + 3 * q + 10 * z)
            if dataset_sizes[q] == 0:
                break

    align_0db_t[c,0], align_3db_t[c,0], ave_perfect_align_rate_t[c,0], ave_achievable_rate_t[c,0], prob_out_perfect[c,0], prob_out[c,0] = \
        HBF_Eva(ns_s, ns_t, data, bf_d, ind_val_test_d, y_val_test_d, snr_out, 'd_r')  # DEACT Method as a Hierarchical Beam Search
    print('DEACT-0db', align_0db_t[c,0])
    print('DEACT-Ach', ave_achievable_rate_t[c, 0])
    print('DEACT-Pout', prob_out[c, 0])
    c += 1
    align_0db_t[c,0], align_3db_t[c,0], ave_perfect_align_rate_t[c,0], ave_achievable_rate_t[c,0], prob_out_perfect[c,0], prob_out[c,0] = \
        HBF_Eva(ns_s, ns_t, data, bf_d, ind_val_test_d, y_val_test_d, snr_out, 'b_r')  # BMW-SS Method as a Hierarchical Beam Search
    print('BMW-SS-0db', align_0db_t[c, 0])
    print('BMW-SS-Ach', ave_achievable_rate_t[c, 0])
    print('BMW-SS-Pout', prob_out[c, 0])
    c += 1
    """
        Plots parameters
    """
    fontsize = 18
    fontweight = 'bold'
    fontproperties = {'size': fontsize}
    if -1 in dataset_sizes:
        dataset_sizes[dataset_sizes.index(-1)] = np.shape(x_train_d)[0]

    T = 200
    a = np.arange(1, T + 1).astype(dtype=np.float64)
    coef = np.repeat(np.reshape((1 - a[top_n - 1] / T), [1, -1]), c-2, axis=0)
    ave_achievable_rate_t_coef = np.zeros_like(ave_achievable_rate_t)
    ave_achievable_rate_t_coef[:c-2, :] = np.multiply(ave_achievable_rate_t[:c-2, :], coef)
    coef_hbs = np.repeat(np.reshape((1 - len_HBS / T), [-1]), 2, axis=0)
    ave_achievable_rate_t_coef[c - 2:c, 0] = np.multiply(ave_achievable_rate_t[c - 2:c, 0], coef_hbs)
    ave_perfect_align_rate_t_coef = np.multiply(ave_perfect_align_rate_t[:c-2, :], coef)
    n_types = len(Fr_list) + 1

    """
        MisAlignment vs N_b
    """
    fig0 = plt.figure(figsize=(14.3,7.7))

    for g in range(n_types):
        plt.plot(top_n, 1 - align_0db_t[g+1], color='k',
                 marker=marker_list[g % len(marker_list)], markersize=7, fillstyle='none')

    lines = []
    for k in range(n_rows):
        if k % n_types == 0:
            if k >= len(counter):
                break
            k_col = int((k + 1) / n_types)
            l, = plt.plot(top_n, 1 - align_0db_t[k], color=color_list[k_col % len(color_list)], fillstyle='none')
            lines.append(l)

    for g in range(2):
        plt.plot(len_HBS, 1 - align_0db_t[c-2+g, 0], color=color_list[0],
                 marker=marker_list[(g+3) % len(marker_list)], markersize=10, fillstyle='none')
    for k in range(n_rows):
        if k >= len(counter):
            break
        if k == 0:
            k_mrk = 1
            k_col = 0
        else:
            k_mrk = (k - 1) % n_types
            k_col = int((k - 1) / n_types) + 1
        plt.plot(top_n, 1 - align_0db_t[k], color=color_list[k_col % len(color_list)],
                 marker=marker_list[k_mrk % len(marker_list)], markersize=7, fillstyle='none')

    arrow_properties = dict(
        facecolor="black", width=0.5,
        headwidth=4, shrink=0.1)

    plt.annotate(
        "DEACT", xy=(len_HBS, 1 - align_0db_t[c - 2, 0]),
        xytext=(len_HBS-8, 1 - align_0db_t[c-2, 0]),  fontsize=18,
        arrowprops=arrow_properties)
    plt.annotate(
        "BMW-SS", xy=(len_HBS, 1 - align_0db_t[c - 1, 0]),
        xytext=(len_HBS+4, 1 - align_0db_t[c - 1, 0]),  fontsize=18,
        arrowprops=arrow_properties)

    plt.yscale('log')
    min_val = 100
    for k in range(n_rows):
        if k >= len(counter):
            break
        min_val = np.amin([min_val, np.amin(1 - align_0db_t[k])])
    min_val = np.amax([min_val, 1e-4])
    l = 10 ** (math.floor(math.log10(min_val)))
    plt.ylim(math.floor(min_val / l) * l, 1)
    plt.ylabel('MisAlignment Probability', fontproperties)
    plt.xlabel('Number of Beam Pairs Scanned', fontproperties)
    ax = gca()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(fontsize)

    leg_list = ['0']
    lines_list = [lines[0]]
    for zz in range(1, len(dataset_sizes)):
        leg_list.append('{}'.format(int(dataset_sizes[zz])))
        lines_list.append(lines[zz])
    if f_ft == 0:
        title_str = "Destination dataset size"
    else:
        title_str = "Target dataset size"
    legend1 = plt.legend(lines_list, leg_list, title="Destination dataset size", loc=1, prop=fontproperties)

    title1 = legend1.get_title()
    title1.set_fontsize(15)
    leg2 = ["DNN", "DNN-TL", "DNN-TL-FR"]
    plt.legend(leg2[:n_types], loc=3, prop=fontproperties)
    plt.gca().add_artist(legend1)

    plt.grid(True, color='#D3D3D3', which='major', linestyle='-')
    plt.grid(True, color='#D3D3D3', which='minor', axis='y', linestyle=':')
    # plt.show(block=True)

    plt.savefig('3DRaytatracing_Results/MisA_{:04d}_{:04d}_.png'.format(src_nr, dst_nr))
    plt.close()

    """ Spectral Efficiency [bps/Hz] 1e5 - Coef"""
    fig0 = plt.figure(figsize=[14.3, 7.7])
    plt.plot(top_n, ave_perfect_align_rate_t_coef[1], 'k--')

    for g in range(n_types):
        plt.plot(top_n, ave_achievable_rate_t_coef[g+1], color='k',
                 marker=marker_list[g % len(marker_list)], markersize=7, fillstyle='none')

    lines = []
    for k in range(n_rows):
        if k % n_types == 0:
            if k >= len(counter):
                break
            k_col = int((k + 1) / n_types)
            l, = plt.plot(top_n, ave_achievable_rate_t_coef[k], color=color_list[k_col % len(color_list)], fillstyle='none')
            lines.append(l)

    for g in range(2):
        plt.plot(len_HBS, ave_achievable_rate_t_coef[c - 2 + g, 0], color=color_list[0],
                 marker=marker_list[(g + 3) % len(marker_list)], markersize=10, fillstyle='none')
    for k in range(n_rows):
        if k >= len(counter):
            break
        if k == 0:
            k_mrk = 1
            k_col = 0
        else:
            k_mrk = (k - 1) % n_types
            k_col = int((k - 1) / n_types) + 1
        plt.plot(top_n, ave_achievable_rate_t_coef[k], color=color_list[k_col % len(color_list)],
                 marker=marker_list[k_mrk % len(marker_list)], markersize=7, fillstyle='none')

    arrow_properties = dict(
        facecolor="black", width=0.5,
        headwidth=4, shrink=0.1)

    plt.annotate(
        "DEACT", xy=(len_HBS, ave_achievable_rate_t_coef[c - 2, 0]),
        xytext=(len_HBS-8 + f_ft*12, ave_achievable_rate_t_coef[c - 2, 0]), fontsize=18,
        arrowprops=arrow_properties)
    plt.annotate(
        "BMW-SS", xy=(len_HBS, ave_achievable_rate_t_coef[c - 1, 0]),
        xytext=(len_HBS-8 , ave_achievable_rate_t_coef[c - 1, 0]), fontsize=18,
        arrowprops=arrow_properties)

    min_val = 100
    for k in range(n_rows):
        if k >= len(counter)-1:
            break
        min_val = np.amin([min_val, np.amin(ave_achievable_rate_t_coef[k])])
    min_val = np.amax([min_val, 1e-3])

    plt.ylim(min_val, np.ceil(np.amax(ave_perfect_align_rate_t_coef[1])))
    plt.xlim(0, top_n[-1]+0.5)
    plt.ylabel('Effective Spectral Efficiency [bps/Hz]', fontproperties)
    plt.xlabel('Number of Beam Pairs Scanned', fontproperties)
    ax = gca()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(fontsize)

    leg_list = ['0']
    lines_list = [lines[0]]
    for zz in range(1, len(dataset_sizes)):
        leg_list.append('{}'.format(int(dataset_sizes[zz])))
        lines_list.append(lines[zz])
    if f_ft == 0:
        title_str = "Destination dataset size"
    else:
        title_str = "Target dataset size"
    legend1 = plt.legend(lines_list,leg_list, title="Destination dataset size", loc=4, prop=fontproperties)
    title1 = legend1.get_title()
    title1.set_fontsize(15)
    leg2 = ["Perfect Alignment", "DNN", "DNN-TL", "DNN-TL-FR"]  # ["Perfect Alignment", "DEACT", "BMW-SS", "DNN", "DNN-TL"]
    plt.legend(leg2[:n_types+1], loc=8, prop=fontproperties)
    plt.gca().add_artist(legend1)

    plt.grid(True, color='#D3D3D3', which='major', linestyle='-')
    plt.grid(True, color='#D3D3D3', which='minor', axis='y', linestyle=':')
    # plt.show(block=True)
    plt.savefig('3DRaytatracing_Results/ESE_{:04d}_{:04d}_.png'.format(src_nr, dst_nr))
    plt.close()