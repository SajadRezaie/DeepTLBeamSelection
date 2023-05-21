import tensorflow as tf
import numpy as np
import h5py, math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os.path


def deep_nn_tmp(ns_s, ns_t, top_n, data, ind_train, x_train, y_train, bf, ind_val, x_val, y_val, ind_test, x_test, y_test,
            m_label, flag_ratio, snr_out, flag_ft=0, flag_freeze=0, flag_freeze_last=0, flag_train=0, load_name=None,
            save_name=None, ns_t_load=None, epochs_batch_list_inp=None):
    if ns_t_load is None:
        ns_t_load = ns_t
    if flag_freeze == 0:
        train_w = True
    else:
        train_w = False
    if flag_freeze_last == 0:
        train_w_last = True
    else:
        train_w_last = False

    """ Define layers of the network """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(np.shape(x_train)[1],), activation=tf.nn.tanh, trainable=train_w),
        tf.keras.layers.Dropout(0.1, trainable=train_w),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh, trainable=train_w),
        tf.keras.layers.Dropout(0.1, trainable=train_w),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh, trainable=train_w),
        tf.keras.layers.Dropout(0.1, trainable=train_w),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh, trainable=train_w),
        tf.keras.layers.Dropout(0.1, trainable=train_w),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, trainable=train_w),
        tf.keras.layers.Dropout(0.1, trainable=train_w),
        tf.keras.layers.Dense(ns_t, activation=tf.nn.softmax, trainable=train_w_last)
    ])
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(np.shape(x_train)[1],), activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=tf.nn.tanh),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(ns_t_load, activation=tf.nn.softmax)
    ])

    if epochs_batch_list_inp is None:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    if load_name is not None:
        if save_name is not None and os.path.isfile(save_name) and flag_train == 0:
            load_name = save_name
            save_name = None
            flag_ft = 0

        if os.path.isfile(load_name) or (save_name is not None):
            if flag_train == 0 or (save_name is not None):
                if flag_ft == 0:
                    model.load_weights(load_name)
                    print('Weights loaded successfully')
                else:
                    model1.load_weights(load_name)
                    g1 = model1.get_weights()
                    model.set_weights(g1[:-2])
                    print('Weights loaded successfully (Not the last layer)')
        else:
            save_name = load_name

    h = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    """ Summary of all the layers """
    # model.summary()
    if (save_name is not None) or flag_train != 0:
        y_ = data[:, bf['ind_s'][0]:bf['ind_s'][0] + m_label]

        ind_s = (data[:, bf['ind_s']]).astype(int)
        rss_2 = data[:, bf['rss']]
        ind_max_rss = np.argsort(-rss_2, axis=1)
        for k_1 in range(np.shape(ind_s)[0]):
            y_[k_1, :] = ind_s[k_1, ind_max_rss[k_1, 0:m_label]]

        y_ = y_.astype(int)
        y_1hot = np.zeros((np.shape(data)[0], ns_t))
        if flag_ratio == 0:
            for z in range(m_label):
                y_1hot[np.arange(np.shape(data)[0]), y_[:, z]] = 1
        else:
            for z in range(m_label):
                y_1hot[np.arange(np.shape(data)[0]), y_[:, z]] = 1/m_label
        y_train = y_1hot[ind_train, :]
        y_val = y_1hot[ind_val, :]

        print('Start training ...')
        if epochs_batch_list_inp is None:
            epochs_batch_list = [(20, 32), (20, 128), (5, 1024), (5, 8192)]
        else:
            epochs_batch_list = epochs_batch_list_inp
        for k in range(len(epochs_batch_list)):
            if epochs_batch_list_inp is not None:
                opt = tf.keras.optimizers.Adam(lr=min(0.00001*(epochs_batch_list[k][1]**2), 0.01))
                model.compile(optimizer=opt,
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            history = model.fit(x_train, y_train, epochs=epochs_batch_list[k][0], batch_size=epochs_batch_list[k][1],
                                validation_data=(x_val, y_val))
            h['acc'].extend(history.history['acc'])
            h['val_acc'].extend(history.history['val_acc'])
            h['loss'].extend(history.history['loss'])
            h['val_loss'].extend(history.history['val_loss'])
        if save_name is None:
            save_name = load_name

        model.save_weights(save_name)

    print('Start processing test data ...')
    predict1 = model.predict(x_test)
    ind_sort = (-predict1).argsort(axis=1)

    print('Start calculating accuracy ...')
    data_test = data[ind_test, :]
    ind_sort_2 = data_test[:, bf['ind_s']]
    rss_2 = data_test[:, bf['rss']]
    snr_2 = data_test[:, bf['snr']]
    align_0db = np.zeros([np.shape(top_n)[0]], dtype=int)
    align_3db = np.zeros([np.shape(top_n)[0]], dtype=int)
    outage_perfect_num = np.zeros([1], dtype=int)
    outage_num = np.zeros([np.shape(top_n)[0]], dtype=int)
    ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    ave_perfect_align_rate = np.zeros([1], dtype=float)
    ave_perfect_snr = np.zeros([1], dtype=float)

    for m in range(np.shape(x_test)[0]):
        ind_s = (ind_sort_2[m, :]).astype(int)
        if 1:  # ns_s != ns_t:
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
        ave_perfect_align_rate[0] += math.log2(1 + snr_max)
        ave_perfect_snr[0] += snr_max
        if snr_max < snr_out:
            outage_perfect_num[0] += 1
        """ 0dB and 3dB power loss probability """
        for k in range(np.shape(top_n)[0]):
            i_max = int(np.argmax(d[ind_sort[m, 0:top_n[k]]], axis=0))
            rss_top_n = d[ind_sort[m, i_max]]
            ave_achievable_rate[k] += math.log2(1 + snr[ind_sort[m, i_max]])
            if rss_top_n >= d[y_test[m]]:
                align_0db[k] += 1
            if rss_top_n > 0.5 * d[y_test[m]]:
                align_3db[k] += 1
            if snr[ind_sort[m, i_max]] < snr_out:
                outage_num[k] += 1

    prob_align_0db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_align_3db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_out_perfect = np.zeros([1], dtype=float)
    prob_out = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_perfect_align_rate = np.zeros([1], dtype=float)
    num_iter = float(np.shape(x_test)[0])
    for k in range(np.shape(top_n)[0]):
        prob_align_0db[k] = align_0db[k] / num_iter
        prob_align_3db[k] = float(align_3db[k]) / num_iter
        prob_ave_achievable_rate[k] = float(ave_achievable_rate[k]) / num_iter
        prob_out[k] = float(outage_num[k]) / (num_iter + 1e-10)
    prob_out_perfect[0] = float(outage_perfect_num[0]) / (num_iter + 1e-10)
    prob_ave_perfect_align_rate[0] = float(ave_perfect_align_rate[0]) / num_iter
    prob_ave_perfect_align_rate_r = np.repeat(prob_ave_perfect_align_rate, np.shape(top_n)[0])
    return h, prob_align_0db, prob_align_3db, prob_ave_perfect_align_rate_r, prob_ave_achievable_rate, \
        prob_out_perfect, prob_out


if __name__ == '__main__':
    ns_s = 100
    ns_t = 64 * 64
    top_n = np.arange(1, 51, 1)

    print('Loading data ...')
    with h5py.File('snr_data_1e5_nt64_nr64_nb_100_so6_mo10.hdf5', 'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data = np.zeros([s[0], s[1]], dtype=float)
        data[:, ] = data_r[:, ]
        print('size:', np.shape(data_r), np.shape(data)[0])
    print('Input data loaded successfully')

    x = np.zeros([np.shape(data)[0], 3])
    x[:, 0] = np.arange(np.shape(data)[0])
    x[:, 1:3] = data[:, :2]
    y = data[:, 2]
    y = y.astype(int)

    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)

    ind_train = x_train[:, 0]
    ind_train = ind_train.astype(int)
    ind_val_test = x_val_test[:, 0]
    ind_val_test = ind_val_test.astype(int)
    ind_val = x_val[:, 0]
    ind_val = ind_val.astype(int)
    ind_test = x_test[:, 0]
    ind_test = ind_test.astype(int)

    x_train = x_train[:, 1:]
    x_val = x_val[:, 1:]
    x_val_test = x_val_test[:, 1:]

    x_test = x_test[:, 1:]

    h, align_0db, align_3db, ave_optimal_rate_r, ave_perfect_align_rate_r, ave_achievable_rate = deep_nn(ns_s, ns_t,
        top_n, data, ind_train, x_train, y_train, ind_val, x_val, y_val, ind_val_test, x_val_test, y_val_test)

    # average 0dB and 3dB Power Loss Probability
    fig1 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(top_n, 1 - align_0db, marker='o')
    plt.plot(top_n, 1 - align_3db, marker='s')
    plt.yscale('log')
    plt.ylim(0.001, 1)
    plt.ylabel('Average Power Loss Probability')
    plt.xlabel('Number of Beam Pairs Scanned')
    plt.legend(['0dB Power Loss', '3dB Power Loss'], loc='upper right')
    plt.grid(True)
    plt.show(block=False)
    # average achievable rate and optimal rate
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(top_n, ave_achievable_rate, marker='o')
    plt.plot(top_n, ave_perfect_align_rate_r, lineStyle='-')
    plt.plot(top_n, ave_optimal_rate_r)
    plt.ylabel('Average Rate (b/s/Hz)')
    plt.xlabel('Number of Beam Pairs Scanned')
    plt.legend(['Achievable Rate', 'Perfect Alignment', 'Optimal'], loc='lower right')
    plt.grid(True)
    plt.show(block=False)

    # summarize history for accuracy
    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(h['acc']); plt.plot(h['val_acc']); plt.title('model accuracy')
    plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(['Train', 'Test'], loc='upper left'); plt.show(block=False)
    # summarize history for loss
    fig4 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(h['loss']); plt.plot(h['val_loss']); plt.title('model loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(['Train', 'Test'], loc='upper right'); plt.show(block=True)