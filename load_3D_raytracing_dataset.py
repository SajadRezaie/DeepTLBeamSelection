import random
import math  # , time, datetime
import numpy as np
#from channel_model import *
import h5py
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle, cmath
from scipy.spatial.transform import Rotation as R
from pylab import *
from matplotlib import colors


def read_raytracing(filename, params):
    """
    Loading the ray tracing file and recording AOD, AOA, ... for all paths of all RX points
    Inputs:
              filename: the filename of the raytracing file
              params: Parameters of the scenario
    Outputs:
              channel_params: list of all paths for all RX points
    """
    with open(filename, 'rb') as f:
        alist = [line.rstrip() for line in f]
    f.close()

    channel_params_all = []
    strating_line = 130

    index_users = []
    pointer = 0
    for l in tqdm(range(strating_line, len(alist))):
        words = alist[l].decode('utf-8').split()
        if len(words) > 0 and words[0] == 'POINT':
            if params['rej_prob'] > 0:
                r = np.random.rand(1)
                if r < params['rej_prob']:
                    continue
            if not params['xlimit'][0] <= float(words[1]) <= params['xlimit'][1]:
                continue
            if not params['ylimit'][0] <= float(words[2]) <= params['ylimit'][1]:
                continue
            if not params['zlimit'][0] <= float(words[3]) <= params['zlimit'][1]:
                continue

            ch_params = dict()
            ch_params['dod_phi'] = []
            ch_params['dod_theta'] = []
            ch_params['doa_phi'] = []
            ch_params['doa_theta'] = []
            ch_params['phase'] = []
            ch_params['toa'] = []
            ch_params['power'] = []
            ch_params['Num_interactions'] = []
            ch_params['Ref_points'] = []
            ch_params['Ref_objs'] = []
            ch_params['Ref_objs_cat'] = []
            ch_params['num_paths'] = 0
            ch_params['loc'] = [float(words[1]), float(words[2]), float(words[3])]
            count = 1
            words_next = alist[l + count].decode('utf-8').split()
            while (len(words_next) > 0 and words_next[0] == 'PATH' and ch_params['num_paths'] < params[
                'max_num_paths']):
                if int(words_next[4]) == 0:
                    diff = np.array(ch_params['loc']) - np.array(params['loc_tx'])
                    dod_phi_LOS = math.atan2(diff[1], diff[0]) % (2 * math.pi)
                    dod_theta_LOS = math.pi / 2 - (math.atan2(diff[2], np.linalg.norm(diff[:2])))
                    doa_phi_LOS = math.atan2(-diff[1], -diff[0]) % (2 * math.pi)
                    doa_theta_LOS = math.pi - dod_theta_LOS
                    ch_params['dod_phi'].append(dod_phi_LOS)
                    ch_params['dod_theta'].append(dod_theta_LOS)
                    ch_params['doa_phi'].append(doa_phi_LOS)
                    ch_params['doa_theta'].append(doa_theta_LOS)
                else:
                    ch_params['dod_phi'].append(float(words_next[8]))
                    ch_params['dod_theta'].append(float(words_next[9]))
                    ch_params['doa_phi'].append(float(words_next[10]))
                    ch_params['doa_theta'].append(float(words_next[11]))
                ch_params['phase'].append(float(words_next[7]))
                ch_params['toa'].append(float(words_next[1]) * 1e-9)  # sec
                ch_params['power'].append(
                    10. ** (.1 * (float(words_next[2]) - 20 * math.log10(params['carrier_freq']) - 167.2)))
                ch_params['Num_interactions'].append(int(words_next[4]))
                ref_p = []
                ref_o = []
                ref_o_cat = ''
                if int(words_next[4]) > 0:
                    for r in range(int(words_next[4])):
                        ref_p.append([float(words_next[12 + r * 6]), float(words_next[13 + r * 6]),
                                      float(words_next[14 + r * 6])])
                        ref_o.append(int(words_next[16 + r * 6]))
                        if int(words_next[16 + r * 6]) == params['objs_num_C']:
                            ref_o_cat += 'C'
                        elif int(words_next[16 + r * 6]) == params['objs_num_F']:
                            ref_o_cat += 'F'
                        else:
                            ref_o_cat += 'W'

                ch_params['Ref_points'].append(ref_p)
                ch_params['Ref_objs'].append(ref_o)
                ch_params['Ref_objs_cat'].append(ref_o_cat)
                ch_params['num_paths'] += 1

                count += 1
                if l + count >= len(alist) - 1:
                    break
                words_next = alist[l + count].decode('utf-8').split()

            channel_params_all.append(ch_params)

            index_users.append(int(pointer))
            pointer += 1
    channel_params = list(channel_params_all[i] for i in index_users)
    return channel_params


def dirglob2loc(phi_g, theta_g, rot):
    """
    Finding the corresponding phi and theta in the local coordinate for a path with phi_g, theta_g in the golbal coordinate
    Inputs:
              phi_g: phi of the path in the global coordinate [rad]
              theta_g: theta of the path in the global coordinate [rad]
              rot: rotation of the local coordinate respect to the global coordinate [rad]
    Outputs:
              phi_l: phi of the path in the local coordinate [rad]
              theta_l: theta of the path in the local coordinate [rad]
    """
    # rot_device : [alpha (around z axis), beta (around y axis), gamma (around x axis)]
    a, b, c = rot[0], rot[1], rot[2]
    theta_l = np.arccos(np.cos(b) * np.cos(c) * np.cos(theta_g) +
                        (np.sin(b) * np.cos(c) * np.cos(phi_g - a) - np.sin(c) * np.sin(phi_g - a)) * np.sin(theta_g))
    re = np.cos(b) * np.sin(theta_g) * np.cos(phi_g - a) - np.sin(b) * np.cos(theta_g)
    im = np.cos(b) * np.sin(c) * np.cos(theta_g) + (
                np.sin(b) * np.sin(c) * np.cos(phi_g - a) + np.cos(c) * np.sin(phi_g - a)) * np.sin(theta_g)
    phi_l = np.arctan2(im, re)
    phi_l = phi_l % (2 * np.pi)
    return phi_l, theta_l


def construct_raytracingmimo_channel(params_user, params, rot_rx):
    """
    Constructing MIMO channel using ray tracing tools
    Inputs:
              params_user: Parameters of the paths for the RX point
              params: General parameters
              rot_rx: rotation of the RX with respect to the global coordinate [rad]
    Outputs:
              channel: The constructed channel
    """
    # rot_rx : [alpha (around z axis), beta (around y axis), gamma (around x axis)]
    kd = 2 * np.pi * params['ant_spacing']
    kd_ms = 2 * np.pi * params['ant_spacing_ms']
    ang_conv = 1
    ts = 1 / params['bw']

    k = np.arange(0, params['ofdm_limit'], params['ofdm_sampling_factor'])
    k = np.reshape(k, (1, np.shape(k)[0]))

    # BS antenna parameters
    mx_ind = np.arange(params['num_ant_x'])
    my_ind = np.arange(params['num_ant_y'])
    mz_ind = np.arange(params['num_ant_z'])
    num_ant = params['num_ant_x'] * params['num_ant_y'] * params['num_ant_z']
    mxx_ind = np.reshape(
        np.repeat(np.reshape(mx_ind, (1, params['num_ant_x'])), params['num_ant_y'] * params['num_ant_z'], axis=0),
        (num_ant, 1))
    myy_ind = np.reshape(
        np.repeat(np.reshape(np.repeat(my_ind, params['num_ant_x']), (1, params['num_ant_x'] * params['num_ant_y'])),
                  params['num_ant_z'], axis=0), (num_ant, 1))
    mzz_ind = np.reshape(np.repeat(mz_ind, params['num_ant_x'] * params['num_ant_y']), (num_ant, 1))
    m = params['num_ant_x'] * params['num_ant_y'] * params['num_ant_z']

    # MS antenna parameters
    mx_ms_ind = np.arange(params['num_ant_ms_x'])
    my_ms_ind = np.arange(params['num_ant_ms_y'])
    mz_ms_ind = np.arange(params['num_ant_ms_z'])
    num_ant_ms = params['num_ant_ms_x'] * params['num_ant_ms_y'] * params['num_ant_ms_z']
    mxx_ms_ind = np.reshape(
        np.repeat(np.reshape(mx_ms_ind, (1, params['num_ant_ms_x'])), params['num_ant_ms_y'] * params['num_ant_ms_z'],
                  axis=0),
        (num_ant_ms, 1))
    myy_ms_ind = np.reshape(
        np.repeat(np.reshape(np.repeat(my_ms_ind, params['num_ant_ms_x']),
                             (1, params['num_ant_ms_x'] * params['num_ant_ms_y'])),
                  params['num_ant_ms_z'], axis=0), (num_ant_ms, 1))
    mzz_ms_ind = np.reshape(np.repeat(mz_ms_ind, params['num_ant_ms_x'] * params['num_ant_ms_y']), (num_ant_ms, 1))
    m_ms = params['num_ant_ms_x'] * params['num_ant_ms_y'] * params['num_ant_ms_z']

    num_sampled_subcarriers = np.shape(k)[1]
    channel = np.zeros((m_ms, m), dtype=complex)

    for l in range(params_user['num_paths']):
        if params_user['Num_interactions'][l] == 0:
            if params['blockage_pr_LoS'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_LoS']:
                    continue
        elif params_user['Num_interactions'][l] == 1:
            if params_user['Ref_objs_cat'][l] == 'C' and params['blockage_pr_RefC'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefC']:
                    continue
            elif params_user['Ref_objs_cat'][l] == 'F' and params['blockage_pr_RefF'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefF']:
                    continue
            elif params_user['Ref_objs_cat'][l] == 'W' and params['blockage_pr_RefW'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefW']:
                    continue
        elif params_user['Num_interactions'][l] == 2:
            if params_user['Ref_objs_cat'][l] == 'WW' and params['blockage_pr_RefWW'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefWW']:
                    continue
            elif (params_user['Ref_objs_cat'][l] == 'WC' or params_user['Ref_objs_cat'][l] == 'CW') and params[
                'blockage_pr_RefWC'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefWC']:
                    continue
            elif (params_user['Ref_objs_cat'][l] == 'FC' or params_user['Ref_objs_cat'][l] == 'CF') and params[
                'blockage_pr_RefCF'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefCF']:
                    continue
            elif (params_user['Ref_objs_cat'][l] == 'FW' or params_user['Ref_objs_cat'][l] == 'WF') and params[
                'blockage_pr_RefWF'] != 0:
                r = np.random.rand(1)
                if r < params['blockage_pr_RefWF']:
                    continue
            else:
                continue
        elif params_user['Num_interactions'][l] > 2:
            continue

        # TX Array
        gamma_x = 1j * kd * np.sin(params_user['dod_theta'][l] * ang_conv) * np.cos(
            params_user['dod_phi'][l] * ang_conv)  # + np.pi/2 (for BS 4)
        gamma_y = 1j * kd * np.sin(params_user['dod_theta'][l] * ang_conv) * np.sin(
            params_user['dod_phi'][l] * ang_conv)
        gamma_z = 1j * kd * np.cos(params_user['dod_theta'][l] * ang_conv)
        gamma_comb = mxx_ind * gamma_x + myy_ind * gamma_y + mzz_ind * gamma_z
        array_response_tx = np.exp(gamma_comb)

        # RX Array
        rot_rx_rad = [i * ang_conv for i in rot_rx]
        phi_doa_l, theta_doa_l = dirglob2loc(params_user['doa_phi'][l] * ang_conv,
                                             params_user['doa_theta'][l] * ang_conv, rot_rx_rad)
        gamma_ms_x = 1j * kd_ms * np.sin(theta_doa_l) * np.cos(phi_doa_l)
        gamma_ms_y = 1j * kd_ms * np.sin(theta_doa_l) * np.sin(phi_doa_l)
        gamma_ms_z = 1j * kd_ms * np.cos(theta_doa_l)
        gamma_comb_ms = mxx_ms_ind * gamma_ms_x + myy_ms_ind * gamma_ms_y + mzz_ms_ind * gamma_ms_z
        array_response_rx = np.exp(gamma_comb_ms)

        delay_normalized = params_user['toa'][l] / ts
        path_const = np.sqrt(params_user['power'][l] / params['num_ofdm']) * np.exp(
            1j * params_user['phase'][l] * ang_conv) * \
                     np.exp(-1j * 2 * np.pi * (k / params['num_ofdm']) * delay_normalized)
        # path_const = 1
        channel_temp = array_response_rx * path_const
        channel += np.matmul(channel_temp, np.reshape(np.conj(np.transpose(array_response_tx)), (1, m)))
        a = 1

    return channel


def hierarchical_3D(h_ch, nt, nr, Kt, Kr, p_tot, N0, ut_list, ur_list, idx_opt=None, rand_seed=None):
    """
    Performing hierarchical beamforming
    Inputs:
              h_ch: The channel
              nt: Number of antennas at TX
              nr: Number of antennas at RX
              Kt: Number of stages in hierarchical search at TX
              Kr: Number of stages in hierarchical search at RX
              p_tot: Transmit power
              N0: The noise power
              ut_list: List of codewords for TX
              ur_list: List of codewords for RX
              idx_opt: index of the best codeword from exhaustive search as a groundtruth
              rand_seed: list of random seeds for each stage of hierarchical search
    Outputs:
              idx_best_tx: Index of the best codeword at TX
              idx_best_rx: Index of the best codeword at RX
              rec_power: Received power using the chosen codewords
              idx_error: Index of the stage that the first mistake happened in the hierarchical search
    """
    best_idx_t_az = []
    best_idx_t_el = []
    best_idx_r_az = []
    best_idx_r_el = []
    ind_t = np.argmax([Kt[0], Kt[1]])
    ind_r = np.argmax([Kr[0], Kr[1]])
    Kt_s = np.sum(Kt)
    Kr_s = np.sum(Kr)
    nt_p = np.prod(nt)
    nr_p = np.prod(nr)
    rec_power = np.zeros([1, Kr_s + Kt_s])
    ut = ut_list[0][0]
    M = 2  # Each parent codeword has M child codewords
    flag_error = 0
    idx_error = np.zeros(Kr_s + Kt_s, dtype=int)
    idx_opt_r = idx_opt % nr_p
    idx_opt_r_el = idx_opt_r % (2 ** Kr[-1])
    idx_opt_r_az = int(math.floor(idx_opt_r / (2 ** Kr[-1])))
    idx_opt_t = int(math.floor(idx_opt / nr_p))
    idx_opt_t_el = idx_opt_t % (2 ** Kt[-1])
    idx_opt_t_az = int(math.floor(idx_opt_t / (2 ** Kt[-1])))
    str_dec = ''
    if Kr[ind_r] > 0:
        str_dec += '0' * (Kr[ind_r] - 1 - max(0, int(np.log2(idx_opt_r_az + 1e-5)))) + bin(int(idx_opt_r_az))[2:]
    if Kr[2] > 0:
        str_dec += '0' * (Kr[-1] - 1 - max(0, int(np.log2(idx_opt_r_el + 1e-5)))) + bin(int(idx_opt_r_el))[2:]
    if Kt[ind_t] > 0:
        str_dec += '0' * (Kt[ind_t] - 1 - max(0, int(np.log2(idx_opt_t_az + 1e-5)))) + bin(int(idx_opt_t_az))[2:]
    if Kt[2] > 0:
        str_dec += '0' * (Kt[-1] - 1 - max(0, int(np.log2(idx_opt_t_el + 1e-5)))) + bin(int(idx_opt_t_el))[2:]
    for k in range(1, Kr_s + 1):
        if k <= Kr[ind_r]:
            k_az = k
            k_el = 0
        else:
            k_az = Kr[ind_r]
            k_el = k - Kr[ind_r]

        """ Codebook Beamforming """
        ur = ur_list[k_az][k_el]
        ur_h = np.conj(np.transpose(ur))
        """ Generate noise """
        if rand_seed is not None:
            np.random.seed(rand_seed[k - 1])

        noise = math.sqrt(N0) * (np.random.normal(0, 1 / math.sqrt(2), [1 * M, nr_p]) +
                                 1j * np.random.normal(0, 1 / math.sqrt(2), [1 * M, nr_p]))
        if k == 1:
            idx_codeword = np.arange(M) * (2 ** k_el)
        elif k_el == 0:
            init = 0
            if best_idx_r_az:
                init = np.sum(M ** np.arange(1, k_az) * np.array(best_idx_r_az))
            idx_codeword = (np.arange(M) + init) * (2 ** k_el)
        elif k_el == 1:
            init = 0
            if best_idx_r_az:
                init = np.sum(M ** np.arange(0, k_az) * np.array(best_idx_r_az))
            idx_codeword = init * (2 ** k_el) + np.arange(M)
        else:
            init = 0
            if best_idx_r_az:
                init = np.sum(M ** np.arange(0, k_az) * np.array(best_idx_r_az))
            idx_codeword = init * (2 ** k_el) + np.arange(M) + np.sum(M ** np.arange(1, k_el) * np.array(best_idx_r_el))
        ut_sel = ut
        ur_h_sel = ur_h[idx_codeword, :]
        noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h_sel, 1, axis=0), noise), axis=1), [1, M])
        """ Received signal strength at each codebook """
        dummy = np.transpose(math.sqrt(p_tot) * np.matmul(np.matmul(ur_h_sel, h_ch), ut_sel))
        s_p = np.square(np.absolute(dummy))
        rss = np.square(np.absolute(dummy + noise_r))
        idx_max = np.argmax(rss)
        if k_el == 0:
            best_idx_r_az.insert(0, idx_max)
        else:
            best_idx_r_el.insert(0, idx_max)
        rec_power[0, k - 1] = 10 * math.log10(rss[0][idx_max])
        if flag_error == 0 and idx_opt is not None:
            if idx_max != int(str_dec[k - 1]):
                flag_error = 1
                idx_error[k - 1] = 1

    if Kr_s == 0:
        idx_best_rx = 0
        ur = ur_list[0][0]
        ur_h = np.conj(np.transpose(ur))
    elif Kr[2] > 0:
        idx_best_rx = idx_codeword[best_idx_r_el[0]]
    else:
        idx_best_rx = idx_codeword[best_idx_r_az[0]]

    ur_h_sel = np.reshape(ur_h[idx_best_rx, :], [-1, nr_p])
    for k in range(1, Kt_s + 1):
        if k <= Kt[ind_t]:
            k_az = k
            k_el = 0
        else:
            k_az = Kt[ind_t]
            k_el = k - Kt[ind_t]
        """ Codebook Beamforming """
        ut = ut_list[k_az][k_el]
        """ Generate noise """
        if rand_seed is not None:
            np.random.seed(rand_seed[Kr_s + k - 1])

        noise = math.sqrt(N0) * (np.random.normal(0, 1 / math.sqrt(2), [M * 1, nr_p]) +
                                 1j * np.random.normal(0, 1 / math.sqrt(2), [M * 1, nr_p]))
        if k == 1:
            idx_codeword = np.arange(M) * (2 ** k_el)
        elif k_el == 0:
            init = 0
            if best_idx_t_az:
                init = np.sum(M ** np.arange(1, k_az) * np.array(best_idx_t_az))
            idx_codeword = (np.arange(M) + init) * (2 ** k_el)
        elif k_el == 1:
            init = 0
            if best_idx_t_az:
                init = np.sum(M ** np.arange(0, k_az) * np.array(best_idx_t_az))
            idx_codeword = init * (2 ** k_el) + np.arange(M)
        else:
            init = 0
            if best_idx_t_az:
                init = np.sum(M ** np.arange(0, k_az) * np.array(best_idx_t_az))
            idx_codeword = init * (2 ** k_el) + np.arange(M) + np.sum(M ** np.arange(1, k_el) * np.array(best_idx_t_el))

        ut_sel = ut[:, idx_codeword]
        noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h_sel, M, axis=0), noise), axis=1), [M, 1])
        """ Received signal strength at each codebook """
        dummy = np.transpose(math.sqrt(p_tot) * np.matmul(np.matmul(ur_h_sel, h_ch), ut_sel))
        s_p = np.square(np.absolute(dummy))
        rss = np.square(np.absolute(dummy + noise_r))
        # print('tx', k, rss, s_p, np.array(aod_list)*180/math.pi)
        idx_max = np.argmax(rss)
        if k_el == 0:
            best_idx_t_az.insert(0, idx_max)
        else:
            best_idx_t_el.insert(0, idx_max)
        rec_power[0, Kr_s + k - 1] = 10 * math.log10(rss[idx_max][0])
        if flag_error == 0 and idx_opt is not None:
            if idx_max != int(str_dec[Kr_s + k - 1]):
                flag_error = 1
                idx_error[Kr_s + k - 1] = 1

    if Kt_s == 0:
        idx_best_tx = 0
    elif Kt[2] > 0:
        idx_best_tx = idx_codeword[best_idx_t_el[0]]
    else:
        idx_best_tx = idx_codeword[best_idx_t_az[0]]
    return idx_best_tx, idx_best_rx, rec_power, idx_error


def hierarchical_3D_Rev(h_ch, nt, nr, Kt, Kr, p_tot, N0, ut_list, ur_list, idx_opt=None, rand_seed=None):
    """
        Performing hierarchical beamforming first in elevation then in azimuth
        Inputs:
                  h_ch: The channel
                  nt: Number of antennas at TX
                  nr: Number of antennas at RX
                  Kt: Number of stages in hierarchical search at TX
                  Kr: Number of stages in hierarchical search at RX
                  p_tot: Transmit power
                  N0: The noise power
                  ut_list: List of codewords for TX
                  ur_list: List of codewords for RX
                  idx_opt: index of the best codeword from exhaustive search as a groundtruth
                  rand_seed: list of random seeds for each stage of hierarchical search
        Outputs:
                  idx_best_tx: Index of the best codeword at TX
                  idx_best_rx: Index of the best codeword at RX
                  rec_power: Received power using the chosen codewords
                  idx_error: Index of the stage that the first mistake happened in the hierarchical search
        """
    best_idx_t_az = []
    best_idx_t_el = []
    best_idx_r_az = []
    best_idx_r_el = []
    ind_t = np.argmax([Kt[0], Kt[1]])
    ind_r = np.argmax([Kr[0], Kr[1]])
    Kt_s = np.sum(Kt)
    Kr_s = np.sum(Kr)
    nt_p = np.prod(nt)
    nr_p = np.prod(nr)
    rec_power = np.zeros([1, Kr_s + Kt_s])
    ut = ut_list[0][0]
    M = 2  # Each parent codeword has M child codewords
    flag_error = 0
    idx_error = np.zeros(Kr_s + Kt_s, dtype=int)
    idx_opt_r = idx_opt % nr_p
    idx_opt_r_el = idx_opt_r % (2 ** Kr[-1])
    idx_opt_r_az = int(math.floor(idx_opt_r / (2 ** Kr[-1])))
    idx_opt_t = int(math.floor(idx_opt / nr_p))
    idx_opt_t_el = idx_opt_t % (2 ** Kt[-1])
    idx_opt_t_az = int(math.floor(idx_opt_t / (2 ** Kt[-1])))
    str_dec = ''
    if Kr[2] > 0:
        str_dec += '0' * (Kr[-1] - 1 - max(0, int(np.log2(idx_opt_r_el + 1e-5)))) + bin(int(idx_opt_r_el))[2:]
    if Kr[ind_r] > 0:
        str_dec += '0' * (Kr[ind_r] - 1 - max(0, int(np.log2(idx_opt_r_az + 1e-5)))) + bin(int(idx_opt_r_az))[2:]
    if Kt[2] > 0:
        str_dec += '0' * (Kt[-1] - 1 - max(0, int(np.log2(idx_opt_t_el + 1e-5)))) + bin(int(idx_opt_t_el))[2:]
    if Kt[ind_t] > 0:
        str_dec += '0' * (Kt[ind_t] - 1 - max(0, int(np.log2(idx_opt_t_az + 1e-5)))) + bin(int(idx_opt_t_az))[2:]
    for k in range(1, Kr_s + 1):
        if k <= Kr[-1]:
            k_az = 0
            k_el = k
        else:
            k_az = k - Kr[-1]
            k_el = Kr[-1]

        """ Codebook Beamforming """
        ur = ur_list[k_az][k_el]
        ur_h = np.conj(np.transpose(ur))
        """ Generate noise """
        if rand_seed is not None:
            np.random.seed(rand_seed[k - 1])

        noise = math.sqrt(N0) * (np.random.normal(0, 1 / math.sqrt(2), [1 * M, nr_p]) +
                                 1j * np.random.normal(0, 1 / math.sqrt(2), [1 * M, nr_p]))

        if k == 1:
            idx_codeword = np.arange(M)
        elif k_az == 0:
            init = 0
            if best_idx_r_el:
                init = np.sum(M ** np.arange(1, k_el) * np.array(best_idx_r_el))
            idx_codeword = np.arange(M) + init
        elif k_az == 1:
            init = 0
            if best_idx_r_el:
                init = np.sum(M ** np.arange(0, k_el) * np.array(best_idx_r_el))
            idx_codeword = init + np.arange(M) * (2 ** k_el)
        else:
            init = 0
            if best_idx_r_el:
                init = np.sum(M ** np.arange(0, k_el) * np.array(best_idx_r_el))
            idx_codeword = init + (np.arange(M) + np.sum(M ** np.arange(1, k_az) * np.array(best_idx_r_az))) * (
                        2 ** k_el)

        ut_sel = ut
        ur_h_sel = ur_h[idx_codeword, :]
        noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h_sel, 1, axis=0), noise), axis=1), [1, M])
        """ Received signal strength at each codebook """
        dummy = np.transpose(math.sqrt(p_tot) * np.matmul(np.matmul(ur_h_sel, h_ch), ut_sel))
        s_p = np.square(np.absolute(dummy))
        rss = np.square(np.absolute(dummy + noise_r))
        # print('rx', k, rss, s_p, np.array(aoa_list)*180/math.pi)
        idx_max = np.argmax(rss)
        if k_az == 0:
            best_idx_r_el.insert(0, idx_max)
        else:
            best_idx_r_az.insert(0, idx_max)
        rec_power[0, k - 1] = 10 * math.log10(rss[0][idx_max])
        if flag_error == 0 and idx_opt is not None:
            if idx_max != int(str_dec[k - 1]):
                flag_error = 1
                idx_error[k - 1] = 1

    if Kr_s == 0:
        idx_best_rx = 0
        ur = ur_list[0][0]
        ur_h = np.conj(np.transpose(ur))
    elif Kr[ind_r] > 0:
        idx_best_rx = idx_codeword[best_idx_r_az[0]]
    else:
        idx_best_rx = idx_codeword[best_idx_r_el[0]]

    ur_h_sel = np.reshape(ur_h[idx_best_rx, :], [-1, nr_p])
    for k in range(1, Kt_s + 1):
        if k <= Kt[-1]:
            k_az = 0
            k_el = k
        else:
            k_az = k - Kt[-1]
            k_el = Kt[-1]
        """ Codebook Beamforming """
        ut = ut_list[k_az][k_el]
        """ Generate noise """
        if rand_seed is not None:
            np.random.seed(rand_seed[Kr_s + k - 1])

        noise = math.sqrt(N0) * (np.random.normal(0, 1 / math.sqrt(2), [M * 1, nr_p]) +
                                 1j * np.random.normal(0, 1 / math.sqrt(2), [M * 1, nr_p]))

        if k == 1:
            idx_codeword = np.arange(M)
        elif k_az == 0:
            init = 0
            if best_idx_t_el:
                init = np.sum(M ** np.arange(1, k_el) * np.array(best_idx_t_el))
            idx_codeword = np.arange(M) + init
        elif k_az == 1:
            init = 0
            if best_idx_t_el:
                init = np.sum(M ** np.arange(0, k_el) * np.array(best_idx_t_el))
            idx_codeword = init + np.arange(M) * (2 ** k_el)
        else:
            init = 0
            if best_idx_t_el:
                init = np.sum(M ** np.arange(0, k_el) * np.array(best_idx_t_el))
            idx_codeword = init + (np.arange(M) + np.sum(M ** np.arange(1, k_az) * np.array(best_idx_t_az))) * (
                        2 ** k_el)

        ut_sel = ut[:, idx_codeword]
        noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h_sel, M, axis=0), noise), axis=1), [M, 1])
        """ Received signal strength at each codebook """
        dummy = np.transpose(math.sqrt(p_tot) * np.matmul(np.matmul(ur_h_sel, h_ch), ut_sel))
        s_p = np.square(np.absolute(dummy))
        rss = np.square(np.absolute(dummy + noise_r))
        # print('tx', k, rss, s_p, np.array(aod_list)*180/math.pi)
        idx_max = np.argmax(rss)
        if k_az == 0:
            best_idx_t_el.insert(0, idx_max)
        else:
            best_idx_t_az.insert(0, idx_max)
        rec_power[0, Kr_s + k - 1] = 10 * math.log10(rss[idx_max][0])
        if flag_error == 0 and idx_opt is not None:
            if idx_max != int(str_dec[Kr_s + k - 1]):
                flag_error = 1
                idx_error[Kr_s + k - 1] = 1

    if Kt_s == 0:
        idx_best_tx = 0
    elif Kt[ind_t] > 0:
        idx_best_tx = idx_codeword[best_idx_t_az[0]]
    else:
        idx_best_tx = idx_codeword[best_idx_t_el[0]]
    return idx_best_tx, idx_best_rx, rec_power, idx_error


def get_response_pos(N, phi):
    """
    Calculating antenna response
    Inputs:
              N: number of antenna elements
              phi: angle of departure (arrivals)
    Output:
              a: antenna response
    """
    a = np.ones([N, 1], dtype=complex)
    sq = math.sqrt(N)
    for k in range(N):
        a[k] = cmath.exp(1j * math.pi * phi * k) / sq
    return a


def create_dictionary_bmw_ss_3D(n_ant, k, antenna_spacing_wavelength_ratio):
    """
    Calculating beam codebook at TX and RX at the kth stage of hierarchical beam alignment
    Inputs:
              n_ant: number of antenna elements at TX/RX
              k: stage of the hierarchical beam alignment
              antenna_spacing_wavelength_ratio: ratio of the antenna spacing to wavelength
    Outputs:
              u: beam codebook at TX/RX
    """
    kd = 2 * np.pi * antenna_spacing_wavelength_ratio

    num_ant = n_ant[0] * n_ant[1] * n_ant[2]
    nb = 2 ** np.array(k)  # number of beams at TX/RX
    num_beams = np.prod(nb)
    u = np.zeros([num_ant, num_beams], dtype=complex)

    mx_ind = np.arange(nb[0])
    my_ind = np.arange(nb[1])
    mz_ind = np.arange(nb[2])

    mxx_ind = np.reshape(np.repeat(np.reshape(mx_ind, (1, nb[0])), nb[1] * nb[2], axis=0), (num_beams, 1))
    myy_ind = np.reshape(np.repeat(np.reshape(np.repeat(my_ind, nb[0]), (1, nb[0] * nb[1])), nb[2], axis=0),
                         (num_beams, 1))
    mzz_ind = np.reshape(np.repeat(mz_ind, nb[0] * nb[1]), (num_beams, 1))
    m_ind = np.concatenate([mxx_ind, myy_ind, mzz_ind], axis=1)

    mx_ind_t = np.arange(n_ant[0])
    my_ind_t = np.arange(n_ant[1])
    mz_ind_t = np.arange(n_ant[2])

    mxx_ind_t = np.reshape(np.repeat(np.reshape(mx_ind_t, (1, n_ant[0])), n_ant[1] * n_ant[2], axis=0), (num_ant, 1))
    myy_ind_t = np.reshape(
        np.repeat(np.reshape(np.repeat(my_ind_t, n_ant[0]), (1, n_ant[0] * n_ant[1])), n_ant[2], axis=0),
        (num_ant, 1))
    mzz_ind_t = np.reshape(np.repeat(mz_ind_t, n_ant[0] * n_ant[1]), (num_ant, 1))
    m_ind_t = np.concatenate([mxx_ind_t, myy_ind_t, mzz_ind_t], axis=1)

    ind_match = np.zeros([num_beams], dtype=int)
    for m in range(num_beams):
        ind_match[m] = np.where((m_ind_t == m_ind[m]).all(axis=1))[0][0]

    ind_az = np.argmax(n_ant[:2])
    nb_az = nb[ind_az]
    if nb[ind_az] == n_ant[ind_az] and nb[2] == n_ant[2]:
        for m in range(nb_az):
            if n_ant[0] > n_ant[1]:
                phi_az = np.arccos((2 * (m + 1) - 1 - nb[0]) / nb[0])
            else:
                phi_az = np.arcsin((2 * (m + 1) - 1 - nb[1]) / nb[1])
            # print(n_ant, k, phi_az * 180 / np.pi)
            for n in range(nb[2]):
                phi_el = np.arccos((2 * (n + 1) - 1 - nb[2]) / (nb[2]))
                gamma_x = 1j * kd * np.sin(phi_el) * np.cos(phi_az)
                gamma_y = 1j * kd * np.sin(phi_el) * np.sin(phi_az)
                gamma_z = 1j * kd * np.cos(phi_el)

                array_response_x = np.exp(mx_ind * gamma_x)
                array_response_y = np.exp(my_ind * gamma_y)
                array_response_az = np.kron(array_response_y, array_response_x)

                array_response_z = np.exp(mz_ind * gamma_z)
                array_response_el = array_response_z
                array_response_K = np.kron(array_response_el, array_response_az)

                u[:, m * nb[2] + n] = array_response_K

    elif nb[2] == n_ant[2]:
        l_az = math.log2(n_ant[ind_az]) - k[ind_az]
        M_az = 2 ** math.floor((l_az + 1) / 2)
        Ns_az = int(n_ant[ind_az] / M_az)

        if l_az % 2 == 0:
            Na_az = M_az
        else:
            Na_az = M_az / 2
        for n in range(nb[2]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nb[2]) / (nb[2]))
            gamma_z = 1j * kd * np.cos(phi_el)
            array_response_z = np.exp(mz_ind * gamma_z)
            array_response_el = array_response_z
            coef_az = np.sin(phi_el)
            coef_phase_az = coef_az # (Ns_az - 1) / Ns_az

            f1 = cmath.exp(-1j * 1 * kd * coef_phase_az) * get_response_pos(Ns_az, coef_az *
                                                                                             (-1 + (2 * 1 - 1) / Ns_az))  # np.sin(phi_el)
            u_k_1 = f1
            for mm in range(2, M_az + 1):
                if mm <= Na_az:
                    fm = cmath.exp(-1j * mm * kd * coef_phase_az) * \
                         get_response_pos(Ns_az, coef_az * (-1 + (2 * mm - 1) / Ns_az))  # np.sin(phi_el)
                else:
                    fm = np.zeros([Ns_az, 1])
                u_k_1 = np.concatenate((u_k_1, fm), axis=0)
            array_response_az = u_k_1[:, 0]
            u_Kron_k_1 = np.kron(array_response_el, array_response_az)

            pow_u_Kron_k_1 = np.sum(np.square(np.abs(u_Kron_k_1)))
            u[:, 0 * nb[2] + n] = u_Kron_k_1 / math.sqrt(pow_u_Kron_k_1)
            for nn in range(2, nb[ind_az] + 1):
                u_k_n = np.multiply(u_k_1, math.sqrt(n_ant[ind_az]) * get_response_pos(n_ant[ind_az], np.sin(phi_el) *
                                                                                       2 * (nn - 1) / (2 ** k[ind_az])))
                array_response_az = u_k_n[:, 0]
                u_Kron_k_n = np.kron(array_response_el, array_response_az)
                pow_u_Kron_k_n = np.sum(np.square(np.abs(u_Kron_k_n)))
                u[:, (nn - 1) * nb[2] + n] = u_Kron_k_n / math.sqrt(pow_u_Kron_k_n)

    elif nb[ind_az] == n_ant[ind_az]:
        l_el = math.log2(n_ant[2]) - k[2]
        M_el = 2 ** math.floor((l_el + 1) / 2)
        Ns_el = int(n_ant[2] / M_el)
        coef_phase_el = 1
        if l_el % 2 == 0:
            Na_el = M_el
        else:
            Na_el = M_el / 2

        for n in range(nb[2]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nb[2]) / nb[2])
            if n == 0:
                f1 = cmath.exp(-1j * 1 * kd * coef_phase_el) * get_response_pos(Ns_el, (-1 + (2 * 1 - 1) / Ns_el))
                u_k_1_el = f1
                for mm in range(2, M_el + 1):
                    if mm <= Na_el:
                        fm = cmath.exp(-1j * mm * kd * coef_phase_el) * get_response_pos(Ns_el,
                                                                                         (-1 + (2 * mm - 1) / Ns_el))
                    else:
                        fm = np.zeros([Ns_el, 1])
                    u_k_1_el = np.concatenate((u_k_1_el, fm), axis=0)
                array_response_el = u_k_1_el[:, 0]
            else:
                u_k_n = np.multiply(u_k_1_el, math.sqrt(n_ant[2]) * get_response_pos(n_ant[2], 2 * n / (2 ** k[2])))
                array_response_el = u_k_n[:, 0]
            for m in range(nb[ind_az]):
                array_response_az = get_response_pos(nb[ind_az],
                                                     np.sin(phi_el) * (-1 + (2 * (m + 1) - 1) / nb[ind_az]))[:, 0]
                u_Kron_k = np.kron(array_response_el, array_response_az)
                pow_u_Kron_k = np.sum(np.square(np.abs(u_Kron_k)))
                u[:, m * nb[2] + n] = u_Kron_k / math.sqrt(pow_u_Kron_k)
    else:
        l_az = math.log2(n_ant[ind_az]) - k[ind_az]
        M_az = 2 ** math.floor((l_az + 1) / 2)
        Ns_az = int(n_ant[ind_az] / M_az)
        if l_az % 2 == 0:
            Na_az = M_az
        else:
            Na_az = M_az / 2

        l_el = math.log2(n_ant[2]) - k[2]
        M_el = 2 ** math.floor((l_el + 1) / 2)
        Ns_el = int(n_ant[2] / M_el)
        if l_el % 2 == 0:
            Na_el = M_el
        else:
            Na_el = M_el / 2

        coef_phase_el = 1
        for n in range(nb[2]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nb[2]) / nb[2])
            if n == 0:
                f1 = cmath.exp(-1j * 1 * kd * coef_phase_el) * get_response_pos(Ns_el, (-1 + (2 * 1 - 1) / Ns_el))
                u_k_1_el = f1
                for mm in range(2, M_el + 1):
                    if mm <= Na_el:
                        fm = cmath.exp(-1j * mm * kd * coef_phase_el) * \
                             get_response_pos(Ns_el, (-1 + (2 * mm - 1) / Ns_el))
                    else:
                        fm = np.zeros([Ns_el, 1])
                    u_k_1_el = np.concatenate((u_k_1_el, fm), axis=0)
                array_response_el = u_k_1_el[:, 0]
            else:
                u_k_n = np.multiply(u_k_1_el, math.sqrt(n_ant[2]) * get_response_pos(n_ant[2], 2 * n / (2 ** k[2])))
                array_response_el = u_k_n[:, 0]

            coef_az = np.sin(phi_el)
            coef_phase_az = coef_az  # (Ns_az - 1) / Ns_az
            f1 = cmath.exp(-1j * 1 * kd * coef_phase_az) * get_response_pos(Ns_az, coef_az *
                                                                                             (-1 + (2 * 1 - 1) / Ns_az))  # np.sin(phi_el)
            u_k_1 = f1
            for mm in range(2, M_az + 1):
                if mm <= Na_az:
                    fm = cmath.exp(-1j * mm * kd * coef_phase_az) * \
                         get_response_pos(Ns_az,coef_az * (-1 + (2 * mm - 1) / Ns_az))  # np.sin(phi_el)
                else:
                    fm = np.zeros([Ns_az, 1])
                u_k_1 = np.concatenate((u_k_1, fm), axis=0)
            array_response_az = u_k_1[:, 0]
            u_Kron_k_1 = np.kron(array_response_el, array_response_az)

            pow_u_Kron_k_1 = np.sum(np.square(np.abs(u_Kron_k_1)))
            u[:, 0 * nb[2] + n] = u_Kron_k_1 / math.sqrt(pow_u_Kron_k_1)
            for nn in range(2, nb[ind_az] + 1):
                u_k_n = np.multiply(u_k_1, math.sqrt(n_ant[ind_az]) * get_response_pos(n_ant[ind_az], np.sin(phi_el) *
                                                                                       2 * (nn - 1) / (2 ** k[ind_az])))
                array_response_az = u_k_n[:, 0]
                u_Kron_k_n = np.kron(array_response_el, array_response_az)
                pow_u_Kron_k_n = np.sum(np.square(np.abs(u_Kron_k_n)))
                u[:, (nn - 1) * nb[2] + n] = u_Kron_k_n / math.sqrt(pow_u_Kron_k_n)

    pow_u = np.sum(np.square(np.abs(u)), axis=0)
    u = u / np.sqrt(pow_u)
    return u


def create_dictionary_deact_3D(n_ant, k, antenna_spacing_wavelength_ratio):
    """
    Calculating beam codebook at TX and RX at the kth stage of hierarchical beam alignment
    Inputs:
              n_ant: number of antenna elements at TX/RX
              k: stage of the hierarchical beam alignment
              antenna_spacing_wavelength_ratio: ratio of the antenna spacing to wavelength
    Outputs:
              u: beam codebook at TX/RX
    """
    kd = 2 * np.pi * antenna_spacing_wavelength_ratio

    num_ant = n_ant[0] * n_ant[1] * n_ant[2]
    nb = 2 ** np.array(k)  # number of beams at TX/RX
    num_beams = np.prod(nb)
    u = np.zeros([num_ant, num_beams], dtype=complex)

    mx_ind = np.arange(nb[0])
    my_ind = np.arange(nb[1])
    mz_ind = np.arange(nb[2])

    mxx_ind = np.reshape(np.repeat(np.reshape(mx_ind, (1, nb[0])), nb[1] * nb[2], axis=0), (num_beams, 1))
    myy_ind = np.reshape(np.repeat(np.reshape(np.repeat(my_ind, nb[0]), (1, nb[0] * nb[1])), nb[2], axis=0),
                         (num_beams, 1))
    mzz_ind = np.reshape(np.repeat(mz_ind, nb[0] * nb[1]), (num_beams, 1))
    m_ind = np.concatenate([mxx_ind, myy_ind, mzz_ind], axis=1)

    mx_ind_t = np.arange(n_ant[0])
    my_ind_t = np.arange(n_ant[1])
    mz_ind_t = np.arange(n_ant[2])

    mxx_ind_t = np.reshape(np.repeat(np.reshape(mx_ind_t, (1, n_ant[0])), n_ant[1] * n_ant[2], axis=0), (num_ant, 1))
    myy_ind_t = np.reshape(
        np.repeat(np.reshape(np.repeat(my_ind_t, n_ant[0]), (1, n_ant[0] * n_ant[1])), n_ant[2], axis=0),
        (num_ant, 1))
    mzz_ind_t = np.reshape(np.repeat(mz_ind_t, n_ant[0] * n_ant[1]), (num_ant, 1))
    m_ind_t = np.concatenate([mxx_ind_t, myy_ind_t, mzz_ind_t], axis=1)

    ind_match = np.zeros([num_beams], dtype=int)
    for m in range(num_beams):
        ind_match[m] = np.squeeze(np.where((m_ind_t == m_ind[m]).all(axis=1)))

    nb_az = max(nb[0], nb[1])
    for m in range(nb_az):
        if n_ant[0] > n_ant[1]:
            phi_az = np.arccos((2 * (m + 1) - 1 - nb[0]) / nb[0])
        else:
            phi_az = np.arcsin((2 * (m + 1) - 1 - nb[1]) / nb[1])
        for n in range(nb[2]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nb[2]) / (nb[2]))

            gamma_x = 1j * kd * np.sin(phi_el) * np.cos(phi_az)
            gamma_y = 1j * kd * np.sin(phi_el) * np.sin(phi_az)
            gamma_z = 1j * kd * np.cos(phi_el)
            gamma_comb = mxx_ind * gamma_x + myy_ind * gamma_y + mzz_ind * gamma_z
            array_response = np.exp(gamma_comb)
            u[ind_match, m * nb[2] + n] = array_response[:, 0]

    pow_u = np.sum(np.square(np.abs(u)), axis=0)
    u = u / np.sqrt(pow_u)
    return u


def create_dictionary(nt, nr, nb_t, nb_r, antenna_spacing_wavelength_ratio):
    """
    Calculating beam codebook at TX and RX - DFT codebooks
    Inputs:
              nt: number of antenna elements at TX
              nr: number of antenna elements at RX
              nb_t: number of beam at TX
              nb_r: number of beam at RX
              antenna_spacing_wavelength_ratio: ratio of the antenna spacing to wavelength
    Outputs:
              ut: beam codebook at TX
              ur: beam codebook at RX
    """
    kd = 2 * np.pi * antenna_spacing_wavelength_ratio

    num_ant_rx = nr[0] * nr[1] * nr[2]
    num_beams_rx = nb_r[0] * nb_r[1]
    ur = np.zeros([num_ant_rx, num_beams_rx], dtype=complex)

    mx_ind_rx = np.arange(nr[0])
    my_ind_rx = np.arange(nr[1])
    mz_ind_rx = np.arange(nr[2])

    mxx_ind_rx = np.reshape(np.repeat(np.reshape(mx_ind_rx, (1, nr[0])), nr[1] * nr[2], axis=0), (num_ant_rx, 1))
    myy_ind_rx = np.reshape(np.repeat(np.reshape(np.repeat(my_ind_rx, nr[0]), (1, nr[0] * nr[1])), nr[2], axis=0),
                            (num_ant_rx, 1))
    mzz_ind_rx = np.reshape(np.repeat(mz_ind_rx, nr[0] * nr[1]), (num_ant_rx, 1))

    for m in range(nb_r[0]):
        if nr[0] > nr[1]:
            phi_az = np.arccos((2 * (m + 1) - 1 - max(nr[0], nr[1])) / max(nr[0], nr[1]))
        else:
            phi_az = np.arcsin((2 * (m + 1) - 1 - max(nr[0], nr[1])) / max(nr[0], nr[1]))
        # print(phi_az*180/np.pi)
        for n in range(nb_r[1]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nr[2]) / (nr[2]))

            gamma_x = 1j * kd * np.sin(phi_el) * np.cos(phi_az)
            gamma_y = 1j * kd * np.sin(phi_el) * np.sin(phi_az)
            gamma_z = 1j * kd * np.cos(phi_el)
            gamma_comb = mxx_ind_rx * gamma_x + myy_ind_rx * gamma_y + mzz_ind_rx * gamma_z
            array_response = np.exp(gamma_comb)
            ur[:, m * nb_r[1] + n] = array_response[:, 0]

    pow_u = np.sum(np.square(np.abs(ur)), axis=0)
    ur = ur / np.sqrt(pow_u)

    num_ant_tx = nt[0] * nt[1] * nt[2]
    num_beams_tx = nb_t[0] * nb_t[1]
    ut = np.zeros([num_ant_tx, num_beams_tx], dtype=complex)

    mx_ind_tx = np.arange(nt[0])
    my_ind_tx = np.arange(nt[1])
    mz_ind_tx = np.arange(nt[2])

    mxx_ind_tx = np.reshape(np.repeat(np.reshape(mx_ind_tx, (1, nt[0])), nt[1] * nt[2], axis=0), (num_ant_tx, 1))
    myy_ind_tx = np.reshape(np.repeat(np.reshape(np.repeat(my_ind_tx, nt[0]), (1, nt[0] * nt[1])), nt[2], axis=0),
                            (num_ant_tx, 1))
    mzz_ind_tx = np.reshape(np.repeat(mz_ind_tx, nt[0] * nt[1]), (num_ant_tx, 1))

    for m in range(nb_t[0]):
        if nt[0] > nt[1]:
            phi_az = np.arccos((2 * (m + 1) - 1 - max(nt[0], nt[1])) / max(nt[0], nt[1]))
        else:
            phi_az = np.arcsin((2 * (m + 1) - 1 - max(nt[0], nt[1])) / max(nt[0], nt[1]))
        # print(phi_az*180/math.pi)
        for n in range(nb_t[1]):
            phi_el = np.arccos((2 * (n + 1) - 1 - nt[2]) / (nt[2]))

            gamma_x = 1j * kd * np.sin(phi_el) * np.cos(phi_az)
            gamma_y = 1j * kd * np.sin(phi_el) * np.sin(phi_az)
            gamma_z = 1j * kd * np.cos(phi_el)
            gamma_comb = mxx_ind_tx * gamma_x + myy_ind_tx * gamma_y + mzz_ind_tx * gamma_z
            array_response = np.exp(gamma_comb)
            ut[:, m * nb_t[1] + n] = array_response[:, 0]

    pow_u = np.sum(np.square(np.abs(ut)), axis=0)
    ut = ut / np.sqrt(pow_u)
    return ut, ur


def raytracingmimo_generator(params):
    """ -------------------------- RaytracingMIMO Dataset Generation ----------------- """
    print(' RaytracingMIMO Dataset Generation started \n')
    cwd = Path.cwd()

    """ Reading ray tracing data """
    print(' Reading the channel parameters of the ray-tracing scenario #s', params['scenario'])

    tx = list()
    for t in range(params['num_bs']):
        filename_data = str(cwd) + '/3D_RayTracing_Scenarios/' + params['scenario'] + '/Site  1 Antenna 1 Rays.str'
        ch_params = read_raytracing(filename_data, params)
        tx.append({'channel_params': ch_params})
    print('Ray tracing datasets are loaded :)')

    # Constructing the channel matrices (Uplink)
    params['nt'] = [params['num_ant_x'], params['num_ant_y'], params['num_ant_z']]
    params['nr'] = [params['num_ant_ms_x'], params['num_ant_ms_y'], params['num_ant_ms_z']]
    n_ant_rx = params['nr'][0] * params['nr'][1] * params['nr'][2]
    n_ant_tx = params['nt'][0] * params['nt'][1] * params['nt'][2]
    params['k_ratio'] = 1
    params['nb_tr'] = [int(n_ant_tx / params['k_ratio']), n_ant_rx]
    params['ns_t'] = int(n_ant_tx / params['k_ratio']) * n_ant_rx
    params['nb_t'] = [max(params['num_ant_x'], params['num_ant_y']), int(params['num_ant_z'] / params['k_ratio'])]
    params['nb_r'] = [max(params['num_ant_ms_x'], params['num_ant_ms_y']), int(params['num_ant_ms_z'] / 1)]
    ut, ur = create_dictionary(params['nt'], params['nr'], params['nb_t'], params['nb_r'], params['ant_spacing'])
    ur_h = np.conj(np.transpose(ur))
    params['ns_s'] = min(100, n_ant_rx * n_ant_tx)

    Kt = [int(math.log2(params['num_ant_x'])), int(math.log2(params['num_ant_y'])),
          int(math.log2(params['num_ant_z']))]  # Number of stages in hierarchical search
    Kr = [int(math.log2(params['num_ant_ms_x'])), int(math.log2(params['num_ant_ms_y'])),
          int(math.log2(params['num_ant_ms_z']))]  # Number of stages in hierarchical search
    ind_t = np.argmax([params['num_ant_x'], params['num_ant_y']])
    ut_deact_list = []
    for k_az in range(Kt[ind_t] + 1):
        ut_deact_list_2 = []
        for k_el in range(Kt[2] + 1):
            k = [0, 0, k_el]
            k[ind_t] = k_az
            ut_deact_list_2.append(create_dictionary_deact_3D(params['nt'], k, params['ant_spacing']))
        ut_deact_list.append(ut_deact_list_2)

    ind_r = np.argmax([params['num_ant_ms_x'], params['num_ant_ms_y']])
    ur_deact_list = []
    for k_az in range(Kr[ind_r] + 1):
        ur_deact_list_2 = []
        for k_el in range(Kr[2] + 1):
            k = [0, 0, k_el]
            k[ind_r] = k_az
            ur_deact_list_2.append(create_dictionary_deact_3D(params['nr'], k, params['ant_spacing']))
        ur_deact_list.append(ur_deact_list_2)

    ut_bmw_ss_list = []
    for k_az in range(Kt[ind_t] + 1):
        ut_bmw_ss_list_2 = []
        for k_el in range(Kt[2] + 1):
            k = [0, 0, k_el]
            k[ind_t] = k_az
            ut_bmw_ss_list_2.append(create_dictionary_bmw_ss_3D(params['nt'], k, params['ant_spacing']))
        ut_bmw_ss_list.append(ut_bmw_ss_list_2)

    ur_bmw_ss_list = []
    for k_az in range(Kr[ind_r] + 1):
        ur_bmw_ss_list_2 = []
        for k_el in range(Kr[2] + 1):
            k = [0, 0, k_el]
            k[ind_r] = k_az
            ur_bmw_ss_list_2.append(create_dictionary_bmw_ss_3D(params['nr'], k, params['ant_spacing']))
        ur_bmw_ss_list.append(ur_bmw_ss_list_2)

    raytracingmimo_dataset = list()
    raytracingmimo_beamforming = list()
    info_tx = list()
    i1 = 12 + params['ns_s']
    i2 = 12 + 2 * params['ns_s']
    i3 = 12 + 3 * params['ns_s']
    beamform_info = {'loc_tx': range(0, 3), 'rot_tx': range(3, 6), 'loc_rx': range(6, 9), 'rot_rx': range(9, 12),
                     'ind_s': range(12, i1), 'rss': range(i1, i2), 'snr': range(i2, i3), 'ave_rss': i3,
                     'ave_snr': i3 + 1,
                     'idx_d': i3 + 2, 'suc_d': i3 + 3, 'rss_d': i3 + 4, 'snr_d': i3 + 5, 'id_err_d': i3 + 6,
                     'idx_d_r': i3 + 7, 'suc_d_r': i3 + 8, 'rss_d_r': i3 + 9, 'snr_d_r': i3 + 10, 'id_err_d_r': i3 + 11,
                     'idx_b': i3 + 12, 'suc_b': i3 + 13, 'rss_b': i3 + 14, 'snr_b': i3 + 15, 'id_err_b': i3 + 16,
                     'idx_b_r': i3 + 17, 'suc_b_r': i3 + 18, 'rss_b_r': i3 + 19, 'snr_b_r': i3 + 20,
                     'id_err_b_r': i3 + 21}
    for t in range(params['num_bs']):
        print('\n Constructing the RaytracingMIMO Dataset for BS #d', t + 1)
        ch_loc_users = list()
        params['num_user'] = len(tx[t]['channel_params'])
        print('num_user', params['num_user'])
        beamform_users = np.zeros([params['num_user'], i3 + 22], dtype=float)
        best_beamform = np.zeros([params['num_user'], 2], dtype=float)
        # noise = math.sqrt(params['noise_p']) * (np.random.normal(0, 1 / math.sqrt(2), [params['ns_t, n_ant_rx]) +
        #                                      1j * np.random.normal(0, 1 / math.sqrt(2), [params['ns_t, n_ant_rx]))
        # noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h, params['nb_tr[0], axis=0), noise), axis=1), params['nb_tr)
        suc_d = 0
        suc_d_rev = 0
        suc_b = 0
        suc_b_rev = 0
        ave_idx_err_d = np.zeros((np.sum(Kr) + np.sum(Kt)))
        ave_idx_err_d_rev = np.zeros((np.sum(Kr) + np.sum(Kt)))
        ave_idx_err_b = np.zeros((np.sum(Kr) + np.sum(Kt)))
        ave_idx_err_b_rev = np.zeros((np.sum(Kr) + np.sum(Kt)))
        ave_rate_d_rev = 0
        ave_rate_b_rev = 0
        num_match = 0
        for user in tqdm(range(0, params['num_user'])):
            alpha_rx = np.random.uniform(0, 2 * math.pi)  # random orientation of the UE (around z-axis) - rad
            beta_rx = np.random.uniform(-math.pi/4, math.pi/4)  # random orientation of the UE (around y-axis) - rad
            gamma_rx = np.random.uniform(-math.pi/4, math.pi/4)  # random orientation of the UE (around x-axis) - rad
            rot_rx = [alpha_rx, beta_rx, gamma_rx]
            channel = construct_raytracingmimo_channel(tx[t]['channel_params'][user], params, rot_rx)
            # channel = np.squeeze(channel)
            loc = tx[t]['channel_params'][user]['loc']

            noise = math.sqrt(params['noise_p']) * (np.random.normal(0, 1 / math.sqrt(2), [params['ns_t'], n_ant_rx]) +
                                                    1j * np.random.normal(0, 1 / math.sqrt(2),
                                                                          [params['ns_t'], n_ant_rx]))
            noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h, params['nb_tr'][0], axis=0), noise), axis=1),
                                 params['nb_tr'])
            if len(np.shape(ur_h)) != 0:
                tmp1 = np.matmul(ur_h, channel)
            else:
                tmp1 = channel
            if len(np.shape(tmp1)) == 1 or np.shape(tmp1)[1] == 1:
                tmp = np.transpose(math.sqrt(params['tp']) * tmp1)
            else:
                tmp = np.transpose(math.sqrt(params['tp']) * np.matmul(tmp1, ut))
            s_p = np.square(np.absolute(tmp))
            rss = np.square(np.absolute(tmp + noise_r))

            snr = s_p / (params['noise_p'] + 1e-40)
            rss_2 = np.ravel(rss)
            snr_2 = np.ravel(snr)

            ind_sort = np.argsort(-snr_2)[:params['ns_s']]
            if np.shape(rss_2)[0] == np.shape(ind_sort)[0]:
                ave_rss = 0
                ave_snr = 0
            else:
                ave_rss = (np.sum(rss_2) - np.sum(rss_2[ind_sort])) / (
                        np.shape(rss_2)[0] - np.shape(ind_sort)[0] + 1e-30)
                ave_snr = (np.sum(snr_2) - np.sum(snr_2[ind_sort])) / (
                        np.shape(snr_2)[0] - np.shape(ind_sort)[0] + 1e-30)

            idx_opt = ind_sort[0]

            if rss_2[idx_opt] == np.amax(rss_2):
                num_match += 1
            print(user+1, num_match, num_match/(user+1))

            rand_seed = np.random.randint(1, 1000, [np.sum(Kr) + np.sum(Kt)])

            """ DEACT """
            idx_best_tx_deact, idx_best_rx_deact, rec_power_m_deact, idx_error_deact = \
                hierarchical_3D(channel, params['nt'], params['nr'], Kt, Kr, params['tp'], params['noise_p'],
                                ut_deact_list, ur_deact_list, idx_opt, rand_seed)

            rss_deact = rss[idx_best_tx_deact, idx_best_rx_deact]
            snr_deact = snr[idx_best_tx_deact, idx_best_rx_deact]
            idx_deact = idx_best_tx_deact * np.prod(params['nr']) + idx_best_rx_deact
            if idx_deact == idx_opt or rss_deact >= rss_2[idx_opt]:
                success_deact = 1
                idx_err_deact = -1
            else:
                success_deact = 0
                idx_err_deact = np.squeeze(np.where(idx_error_deact))

            suc_d += success_deact
            print('\n suc_deact: ', user + 1, suc_d / (user + 1))
            ave_idx_err_d += idx_error_deact
            # print('\n ave_idx_err_d: ', ave_idx_err_d, np.sum(ave_idx_err_d))

            """ DEACT - Reverse"""
            idx_best_tx_deact_rev, idx_best_rx_deact_rev, rec_power_m_deact_rev, idx_error_deact_rev = \
                hierarchical_3D_Rev(channel, params['nt'], params['nr'], Kt, Kr, params['tp'], params['noise_p'],
                                    ut_deact_list, ur_deact_list, idx_opt, rand_seed)

            rss_deact_rev = rss[idx_best_tx_deact_rev, idx_best_rx_deact_rev]
            snr_deact_rev = snr[idx_best_tx_deact_rev, idx_best_rx_deact_rev]
            idx_deact_rev = idx_best_tx_deact_rev * np.prod(params['nr']) + idx_best_rx_deact_rev
            if idx_deact_rev == idx_opt or rss_deact_rev >= rss_2[idx_opt]:
                success_deact_rev = 1
                idx_err_deact_rev = -1
            else:
                success_deact_rev = 0
                idx_err_deact_rev = np.squeeze(np.where(idx_error_deact_rev))

            suc_d_rev += success_deact_rev
            ave_rate_d_rev += math.log2(1 + snr_deact_rev)
            print('\n suc_deact_rev: ', user + 1, suc_d_rev / (user + 1))
            ave_idx_err_d_rev += idx_error_deact_rev
            print('\n ave_idx_err_d_rev: ', ave_idx_err_d_rev, np.sum(ave_idx_err_d_rev), ave_rate_d_rev / (user + 1))

            """ BMW_SS """
            idx_best_tx_bmw_ss, idx_best_rx_bmw_ss, rec_power_m_bmw_ss, idx_error_bmw_ss = \
                hierarchical_3D(channel, params['nt'], params['nr'], Kt, Kr, params['tp'], params['noise_p'],
                                ut_bmw_ss_list, ur_bmw_ss_list, idx_opt, rand_seed)

            rss_bmw_ss = rss[idx_best_tx_bmw_ss, idx_best_rx_bmw_ss]
            snr_bmw_ss = snr[idx_best_tx_bmw_ss, idx_best_rx_bmw_ss]
            idx_bmw_ss = idx_best_tx_bmw_ss * np.prod(params['nr']) + idx_best_rx_bmw_ss
            if idx_bmw_ss == idx_opt or rss_bmw_ss >= rss_2[idx_opt]:
                success_bmw_ss = 1
                idx_err_bmw_ss = -1
            else:
                success_bmw_ss = 0
                idx_err_bmw_ss = np.squeeze(np.where(idx_error_bmw_ss))

            suc_b += success_bmw_ss
            print('\n suc_bmw_ss: ', user + 1, suc_b / (user + 1))
            ave_idx_err_b += idx_error_bmw_ss
            # print('\n ave_idx_err_b: ', ave_idx_err_b, np.sum(ave_idx_err_b))

            """ BMW_SS - Reverse"""
            idx_best_tx_bmw_ss_rev, idx_best_rx_bmw_ss_rev, rec_power_m_bmw_ss_rev, idx_error_bmw_ss_rev = \
                hierarchical_3D_Rev(channel, params['nt'], params['nr'], Kt, Kr, params['tp'], params['noise_p'],
                                    ut_bmw_ss_list, ur_bmw_ss_list, idx_opt, rand_seed)
            rss_bmw_ss_rev = rss[idx_best_tx_bmw_ss_rev, idx_best_rx_bmw_ss_rev]
            snr_bmw_ss_rev = snr[idx_best_tx_bmw_ss_rev, idx_best_rx_bmw_ss_rev]
            idx_bmw_ss_rev = idx_best_tx_bmw_ss_rev * np.prod(params['nr']) + idx_best_rx_bmw_ss_rev
            if idx_bmw_ss_rev == idx_opt or rss_bmw_ss_rev >= rss_2[idx_opt]:
                success_bmw_ss_rev = 1
                idx_err_bmw_ss_rev = -1
            else:
                success_bmw_ss_rev = 0
                idx_err_bmw_ss_rev = np.squeeze(np.where(idx_error_bmw_ss_rev))

            suc_b_rev += success_bmw_ss_rev
            ave_rate_b_rev += math.log2(1 + snr_bmw_ss_rev)
            print('\n suc_bmw_ss_rev: ', user + 1, suc_b_rev / (user + 1))
            ave_idx_err_b_rev += idx_error_bmw_ss_rev
            print('\n ave_idx_err_b_rev: ', ave_idx_err_b_rev, np.sum(ave_idx_err_b_rev), ave_rate_b_rev / (user + 1))

            # ch_loc_users.append({'channel': channel, 'loc': loc})
            beamform_users[user, beamform_info['loc_tx']] = params['loc_tx']  # loc tx
            beamform_users[user, beamform_info['rot_tx']] = params['rot_tx']  # rot tx
            beamform_users[user, beamform_info['loc_rx']] = loc  # loc rx
            beamform_users[user, beamform_info['rot_rx']] = rot_rx  # rot rx
            beamform_users[user, beamform_info['ind_s']] = ind_sort
            beamform_users[user, beamform_info['rss']] = rss_2[ind_sort]
            beamform_users[user, beamform_info['snr']] = snr_2[ind_sort]
            beamform_users[user, beamform_info['ave_rss']] = ave_rss
            beamform_users[user, beamform_info['ave_snr']] = ave_snr

            beamform_users[user, beamform_info['idx_d']] = idx_deact
            beamform_users[user, beamform_info['suc_d']] = success_deact
            beamform_users[user, beamform_info['rss_d']] = rss_deact
            beamform_users[user, beamform_info['snr_d']] = snr_deact
            beamform_users[user, beamform_info['id_err_d']] = idx_err_deact

            beamform_users[user, beamform_info['idx_d_r']] = idx_deact_rev
            beamform_users[user, beamform_info['suc_d_r']] = success_deact_rev
            beamform_users[user, beamform_info['rss_d_r']] = rss_deact_rev
            beamform_users[user, beamform_info['snr_d_r']] = snr_deact_rev
            beamform_users[user, beamform_info['id_err_d_r']] = idx_err_deact_rev

            beamform_users[user, beamform_info['idx_b']] = idx_bmw_ss
            beamform_users[user, beamform_info['suc_b']] = success_bmw_ss
            beamform_users[user, beamform_info['rss_b']] = rss_bmw_ss
            beamform_users[user, beamform_info['snr_b']] = snr_bmw_ss
            beamform_users[user, beamform_info['id_err_b']] = idx_err_bmw_ss

            beamform_users[user, beamform_info['idx_b_r']] = idx_bmw_ss_rev
            beamform_users[user, beamform_info['suc_b_r']] = success_bmw_ss_rev
            beamform_users[user, beamform_info['rss_b_r']] = rss_bmw_ss_rev
            beamform_users[user, beamform_info['snr_b_r']] = snr_bmw_ss_rev
            beamform_users[user, beamform_info['id_err_b_r']] = idx_err_bmw_ss_rev

            best_beamform[user, 0] = ind_sort[0] % params['nb_r'][1]
            best_beamform[user, 1] = np.floor(ind_sort[0] / params['nb_r'][1])

        inf = {'params': params, 'beamform_info': beamform_info}
        info_tx.append(inf)

        # raytracingmimo_dataset.append(ch_loc_users)
        raytracingmimo_beamforming.append(beamform_users)

        # the histogram of the data
        fig1 = plt.figure()
        n, bins, patches = plt.hist(beamform_users[:, 12], params['nb_tr'][0] * params['nb_tr'][1], facecolor='g')
        plt.xlabel('beam index')
        plt.ylabel('Probability')
        plt.title('Frequency of best beam index')
        plt.grid(True)
        plt.show(block=False)

        fig1 = plt.figure()
        n, bins, patches = plt.hist((beamform_users[:, 12] % params['nb_tr'][1]), params['nb_tr'][1], facecolor='g')
        plt.xlabel('beam index')
        plt.ylabel('Probability')
        plt.title('Frequency of best beam index')
        plt.grid(True)
        plt.show(block=False)

        fig1 = plt.figure()
        n, bins, patches = plt.hist(np.floor(beamform_users[:, 12] / params['nb_tr'][1]), params['nb_tr'][0],
                                    facecolor='g')
        plt.xlabel('beam index')
        plt.ylabel('Probability')
        plt.title('Frequency of best beam index')
        plt.grid(True)
        plt.show(block=False)

    print('\n RaytracingMIMO Dataset Generation completed \n')
    # Save Dataset
    pickle.dump(info_tx, open("3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.p".format(save_id), "wb"))
    print('Parameter file is saved :)')
    with h5py.File('3DRaytatracing_Datasets/raytracingmimo_data_{:04d}.hdf5'.format(save_id), 'w') as f:
        dset = f.create_dataset("default", data=raytracingmimo_beamforming)
    print('dataset is saved :)', '  save_id', save_id)
    return raytracingmimo_dataset


if __name__ == '__main__':
    save_id = 60
    print('save_id', save_id)
    # ------  Inputs to the RaytracingMIMO dataset generation code ------------ #
    # ------Ray-tracing scenario
    params = {}
    params['scenario'] = 'LR'  # The adopted ray tracing scenarios
    params['carrier_freq'] = 60  # GHz
    params['num_bs'] = 1
    # ------MIMO parameters set

    # Active users

    params['xlimit'] = [-10, 10]
    params['ylimit'] = [-10, 10]
    params['zlimit'] = [-10, 10]

    # Number of BS Antenna (TX)
    params['num_ant_x'] = 1  # Number of the UPA antenna array on the x-axis
    params['num_ant_y'] = 64  # Number of the UPA antenna array on the y-axis
    params['num_ant_z'] = 1  # Number of the UPA antenna array on the z-axis

    # Number of MS Antenna (RX)
    params['num_ant_ms_x'] = 1  # Number of the UPA antenna array on the x-axis
    params['num_ant_ms_y'] = 16  # Number of the UPA antenna array on the y-axis
    params['num_ant_ms_z'] = 1  # Number of the UPA antenna array on the z-axis

    # Antenna spacing
    params['ant_spacing'] = .5  # ratio of the wavelength for half wavelength enter .5
    params['ant_spacing_ms'] = .5  # ratio of the wavelength for half wavelength enter .5

    # System bandwidth
    params['bandwidth'] = 1  # The bandiwdth in GHz

    # ofdm parameters
    params['num_ofdm'] = 1  # Number of ofdm subcarriers
    params['ofdm_sampling_factor'] = 1  # The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
    params['ofdm_limit'] = 1  # Only the first params['ofdm_limit'] subcarriers will be considered when constructing the channels

    # Number of paths
    params['max_num_paths'] = 25  # Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path

    # -------------------------- RaytracingMIMO Dataset Generation -----------------#
    if params['scenario'] == 'AC' or params['scenario'] == 'AC-2' or params['scenario'] == 'AC-3':
        params['loc_tx'] = [3.5, 4.5, 2.5]
        params['rot_tx'] = [0, 0, 0]
        params['rej_prob'] = 0.9
        params['blockage_pr_LoS'] = 0
        params['blockage_pr_RefW'] = 0
        params['blockage_pr_RefC'] = 0
        params['blockage_pr_RefF'] = 0
        params['blockage_pr_RefWW'] = 0
        params['blockage_pr_RefWC'] = 0
        params['blockage_pr_RefCF'] = 0
        params['blockage_pr_RefWF'] = 0
        params['objs_num_C'] = 0
        params['objs_num_F'] = 0
    elif params['scenario'] == 'CR-AP':
        params['loc_tx'] = [4, 1.5, 2.9]
        params['rot_tx'] = [0, 0, 0]
        params['rej_prob'] = 0
        params['blockage_pr_LoS'] = 0.5
        params['blockage_pr_RefW'] = 0.126
        params['blockage_pr_RefC'] = 1
        params['blockage_pr_RefF'] = 1
        params['blockage_pr_RefWW'] = 0.1449  # (p + p + p**2)
        params['blockage_pr_RefWC'] = 1
        params['blockage_pr_RefCF'] = 1
        params['blockage_pr_RefWF'] = 1
        params['objs_num_C'] = 12
        params['objs_num_F'] = 1
    elif params['scenario'] == 'LR':
        params['loc_tx'] = [7, 3.5, 1.5]
        params['rot_tx'] = [0, 0, 0]
        params['rej_prob'] = 0
        params['blockage_pr_LoS'] = 0.5
        params['blockage_pr_RefW'] = 0.4
        params['blockage_pr_RefC'] = 0
        params['blockage_pr_RefF'] = 0.7
        params['blockage_pr_RefWW'] = 0.8
        params['blockage_pr_RefWC'] = 0.3
        params['blockage_pr_RefCF'] = 0.8
        params['blockage_pr_RefWF'] = 0.7
        params['objs_num_C'] = 5
        params['objs_num_F'] = 1

    # Calculate transmitter power and noise power
    params['tp_dbm'] = 0  # power of transmitter (dbm)
    params['bw'] = params['bandwidth'] * 1e9  # Bandwidth in Hz
    params['n0'] = -174  # Noise spectral density (dBm)
    params['nf'] = 0  # Noise figure (dB)
    params['tp'] = 10 ** (params['tp_dbm'] / 10)  # power of transmitter (mW)
    params['noise_p_dbm'] = params['n0'] + 10 * math.log10(params['bw']) + params[
        'nf']  # power of noise at receiver (dBm)
    params['noise_p'] = 10 ** (params['noise_p_dbm'] / 10)  # power of noise at receiver (mW)
    RaytracingMIMO_dataset = raytracingmimo_generator(params)


