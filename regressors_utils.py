import numpy as np
import pickle
import mne
from signal_utils import sparse_resample
import os
from os.path import join


def get_regressors(fs = 100, 
                   acoustic_path = 'C:/Users/D-CAP/Documents/GitHub/witching-star/regressors/selected_regs.pkl',
                   path_temporal = 'C:/Users/D-CAP/Documents/GitHub/witching-star', 
                   temporal_type = 'empyrhytm', folder_version = 'gen22', temporal_fs = 100, shift_value = 5,
                   path_semantic = r"C:/Users/D-CAP/Documents/GitHub/witching-star/semantic_regressor/predictive_regressors_wd.pkl",
                   path_phonemic = r"C:/Users/D-CAP/Documents/GitHub/witching-star/semantic_regressor/phonemic_regressors.pickle",
                   path_syllabic = r"C:/Users/D-CAP/Documents/GitHub/witching-star/semantic_regressor/syllabic_regressors.pickle"):

    # Acoustic Regressors
    acoustic_data = pickle.load(open(acoustic_path, 'rb'))
    data_fs = acoustic_data['fs']
    acoustic_regressors = acoustic_data['regs']
    acoustic_names = acoustic_data['regs_name']
    ratio = fs/data_fs
    duration = int(acoustic_regressors.shape[0] * ratio) 

    acoustic_resamp = []
    for i in range(acoustic_regressors.shape[1]):
        name = acoustic_names[i]
        if name in ['Intensity', 'Envelope Oganian', 'Envelope Derivative TF', 'F0 Loudness', 'SpectralFlux Filtered', 'SpectralFlux not_filtered']:
            new_reg = mne.filter.resample(acoustic_regressors[:,i], up=fs, down=data_fs)[:duration]
        elif name in ['peakEnv_tf', 'Syllabe Onset', 'p-syl', 'Phono']:
            new_reg = acoustic_regressors[:,i] - np.min(acoustic_regressors[:,i])
            new_reg = sparse_resample(new_reg, new_fs = fs, current_fs = data_fs)[:duration]
        else:
            print('wut')
        acoustic_resamp.append(new_reg)
    acoustic_resamp = np.asarray(acoustic_resamp).T
    regressors = acoustic_resamp
    regressors_name = acoustic_names


    # Temporal Surprise Regressors
    path_empyrhythm = os.path.join('C:/Users/D-CAP/Documents/GitHub/witching-star', temporal_type, folder_version)
    regressors_witch = pickle.load(open(join(path_empyrhythm,'witch_regressors.pkl'),'rb'))


    regressor_syl_proba = np.roll(regressors_witch['syl'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_syl_surprise = np.roll(regressors_witch['syl_surprise'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_syl_entropy = np.roll(regressors_witch['syl_entropy'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_syl_ref = np.roll(regressors_witch['syl_ref'][:regressors.shape[0],np.newaxis],shift_value)

    regressor_phn_proba = np.roll(regressors_witch['phn'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_phn_surprise = np.roll(regressors_witch['phn_surprise'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_phn_entropy = np.roll(regressors_witch['phn_entropy'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_phn_ref = np.roll(regressors_witch['phn_ref'][:regressors.shape[0],np.newaxis],shift_value)

    regressor_wrd_proba = np.roll(regressors_witch['wrd'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_wrd_surprise = np.roll(regressors_witch['wrd_surprise'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_wrd_entropy = np.roll(regressors_witch['wrd_entropy'][:regressors.shape[0],np.newaxis],shift_value)
    regressor_wrd_ref = np.roll(regressors_witch['wrd_ref'][:regressors.shape[0],np.newaxis],shift_value)

    temporal_regressors = np.hstack([regressor_phn_surprise,regressor_phn_entropy,
                                regressor_syl_surprise,regressor_syl_entropy,
                                regressor_wrd_surprise,regressor_wrd_entropy])
    
    
    ratio = fs/temporal_fs
    temporal_resamp = []
    for i in range(temporal_regressors.shape[1]):
        new_reg = temporal_regressors[:,i] - np.min(temporal_regressors[:,i])
        new_reg = sparse_resample(new_reg, new_fs = fs, current_fs = temporal_fs)[:duration]
        temporal_resamp.append(new_reg)
    temporal_resamp = np.asarray(temporal_resamp).T
    temporal_regressors = temporal_resamp

    regressors = np.hstack([regressors, temporal_regressors])
    regressors_name = regressors_name + ['phn_temporal_surprise', 'phn_temporal_entropy','syl_temporal_surprise', 'syl_temporal_entropy','wrd_temporal_surprise', 'wrd_temporal_entropy']

    
    # Semantic Regressor
    semantic_data = pickle.load(open(path_semantic, 'rb'))
    semantic_regressors = np.roll(semantic_data['X'][:,0,[2,1]],4,axis=0)[:regressors.shape[0],:]
    semantic_fs = semantic_data['fs']
    ratio = fs/semantic_fs

    sur = semantic_regressors[:,0]
    ent = semantic_regressors[:,1]
    loc_list = []
    sur_list = []
    ent_list = []
    for l,s,e in zip(np.arange(len(sur)),sur,ent):
        if s > 0:
            loc_list.append(l)
            sur_list.append(s)
            ent_list.append(e)

    diff = np.asarray(sur_list)[1:] - np.asarray(ent_list)[:-1]
    add_sur = np.zeros([semantic_regressors.shape[0],1])
    shift_ent = np.zeros([semantic_regressors.shape[0],1])
    shift2_ent = np.zeros([semantic_regressors.shape[0],1])
    shift3_ent = np.zeros([semantic_regressors.shape[0],1])
    shiftp_ent = np.zeros([semantic_regressors.shape[0],1])
    control = np.zeros([semantic_regressors.shape[0],1])
    for l,d in zip(loc_list[1:], diff):
        add_sur[l] = np.abs(d)
        control[l] = 1
    for l,e in zip(loc_list[1:], ent_list[:-1]):
        shift_ent[l] = e
    for l,e in zip(loc_list[2:], ent_list[:-2]):
        shift2_ent[l] = e
    for l,e in zip(loc_list[:-1], ent_list[1:]):
        shiftp_ent[l] = e


    semantic_regressors = np.hstack([semantic_regressors, add_sur, shift_ent, shift2_ent, shiftp_ent, control])
    semantic_resamp = []
    for i in range(semantic_regressors.shape[1]):
        new_reg = semantic_regressors[:,i] - np.min(semantic_regressors[:,i])
        new_reg = sparse_resample(new_reg, new_fs = fs, current_fs = semantic_fs)[:duration]
        semantic_resamp.append(new_reg)
    semantic_resamp = np.asarray(semantic_resamp).T
    semantic_regressors = semantic_resamp

    sur = semantic_regressors[:,0]
    ent = semantic_regressors[:,1]
    tsur = regressors[:,14]
    median_value = np.median(tsur[tsur > 0])
    low_tempsurprise_semantic = sur.copy()
    high_tempsurprise_semantic = sur.copy()
    low_tempsurprise_semantic[tsur > median_value] = 0
    high_tempsurprise_semantic[tsur < median_value] = 0
    low_tempsurprise_semantic = low_tempsurprise_semantic[:,np.newaxis]
    high_tempsurprise_semantic = high_tempsurprise_semantic[:,np.newaxis]

    regressors = np.hstack([regressors, semantic_regressors[:,0:2], low_tempsurprise_semantic, high_tempsurprise_semantic, semantic_regressors[:,2:]])
    regressors_name = regressors_name + ['wrd_semsurprise', 'wrd_sementropy', 'lowtemp_sem', 'hightemp_sem', 'scandal', 'entropy t-1', 'entropy t-2', 'entropy t+1', 'wrd_onset']


    # Phonemic Regressors
    
    phonemic_data = pickle.load(open(path_phonemic, 'rb'))
    data_fs = phonemic_data['fs']
    phonemic_regressor = phonemic_data['X']
    phn_resamp = []
    for i in range(phonemic_regressor.shape[1]):
        new_reg = phonemic_regressor[:,i] - np.min(phonemic_regressor[:,i])
        new_reg = sparse_resample(new_reg, new_fs = fs, current_fs = data_fs)[:regressors.shape[0]]
        phn_resamp.append(new_reg)
    phn_resamp = np.asarray(phn_resamp).T
    phn_resamp = np.roll(phn_resamp,4,axis=0)
    phn_resamp[:,3] = phn_resamp[:,1] - phn_resamp[:,2]
    phn_regressors = phn_resamp

    sur = phn_regressors[:,1]
    ent = phn_regressors[:,2]
    loc_list = []
    sur_list = []
    ent_list = []
    for l,s,e in zip(np.arange(len(sur)),sur,ent):
        if s > 0:
            loc_list.append(l)
            sur_list.append(s)
            ent_list.append(e)

    diff = np.asarray(sur_list)[1:] - np.asarray(ent_list)[:-1]
    add_sur = np.zeros([phn_regressors.shape[0],1])
    shift_ent = np.zeros([phn_regressors.shape[0],1])
    shift2_ent = np.zeros([phn_regressors.shape[0],1])
    shift3_ent = np.zeros([phn_regressors.shape[0],1])
    shiftp_ent = np.zeros([phn_regressors.shape[0],1])
    control = np.zeros([phn_regressors.shape[0],1])
    for l,d in zip(loc_list[1:], diff):
        add_sur[l] = np.abs(d)
        control[l] = 1
    for l,e in zip(loc_list[1:], ent_list[:-1]):
        shift_ent[l] = e
    for l,e in zip(loc_list[2:], ent_list[:-2]):
        shift2_ent[l] = e
    for l,e in zip(loc_list[:-1], ent_list[1:]):
        shiftp_ent[l] = e

    tsur = regressors[:,10]
    median_value = np.median(tsur[tsur > 0])
    low_tempsurprise_phn = sur.copy()
    high_tempsurprise_phn = sur.copy()
    low_tempsurprise_phn[tsur > median_value] = 0
    high_tempsurprise_phn[tsur < median_value] = 0
    low_tempsurprise_phn = low_tempsurprise_phn[:,np.newaxis]
    high_tempsurprise_phn = high_tempsurprise_phn[:,np.newaxis]

    regressors = np.hstack([regressors,phn_regressors,low_tempsurprise_phn,high_tempsurprise_phn,add_sur,shift_ent,shift2_ent,shiftp_ent ])
    regressors_name = regressors_name + ['phn_' + name for name in phonemic_data['names']] + ['phn_lowtemp_sem', 'phn_hightemp_sem', 'phn_scandal', 'phn_entropy t-1', 'phn_entropy t-2', 'phn_entropy t+1']



    # Syllabic Regressors
    
    syllabic_data = pickle.load(open(path_syllabic, 'rb'))
    data_fs = syllabic_data['fs']
    syl_regressor = syllabic_data['X']
    syl_resamp = []
    for i in range(syl_regressor.shape[1]):
        new_reg = syl_regressor[:,i] - np.min(syl_regressor[:,i])
        new_reg = sparse_resample(new_reg, new_fs = fs, current_fs = data_fs)[:regressors.shape[0]]
        syl_resamp.append(new_reg)
    syl_resamp = np.asarray(syl_resamp).T
    syl_resamp = np.roll(syl_resamp,4,axis=0)
    syl_resamp[:,3] = syl_resamp[:,1] - syl_resamp[:,2]
    syl_regressors = syl_resamp

    sur = syl_regressors[:,1]
    ent = syl_regressors[:,2]
    loc_list = []
    sur_list = []
    ent_list = []
    for l,s,e in zip(np.arange(len(sur)),sur,ent):
        if s > 0:
            loc_list.append(l)
            sur_list.append(s)
            ent_list.append(e)

    diff = np.asarray(sur_list)[1:] - np.asarray(ent_list)[:-1]
    add_sur = np.zeros([syl_regressors.shape[0],1])
    shift_ent = np.zeros([syl_regressors.shape[0],1])
    shift2_ent = np.zeros([syl_regressors.shape[0],1])
    shift3_ent = np.zeros([syl_regressors.shape[0],1])
    shiftp_ent = np.zeros([syl_regressors.shape[0],1])
    control = np.zeros([syl_regressors.shape[0],1])
    for l,d in zip(loc_list[1:], diff):
        add_sur[l] = np.abs(d)
        control[l] = 1
    for l,e in zip(loc_list[1:], ent_list[:-1]):
        shift_ent[l] = e
    for l,e in zip(loc_list[2:], ent_list[:-2]):
        shift2_ent[l] = e
    for l,e in zip(loc_list[:-1], ent_list[1:]):
        shiftp_ent[l] = e

    tsur = regressors[:,12]
    median_value = np.median(tsur[tsur > 0])
    low_tempsurprise_syl = sur.copy()
    high_tempsurprise_syl = sur.copy()
    low_tempsurprise_syl[tsur > median_value] = 0
    high_tempsurprise_syl[tsur < median_value] = 0
    low_tempsurprise_syl = low_tempsurprise_syl[:,np.newaxis]
    high_tempsurprise_syl = high_tempsurprise_syl[:,np.newaxis]

    regressors = np.hstack([regressors,syl_regressors,low_tempsurprise_syl,high_tempsurprise_syl,add_sur,shift_ent,shift2_ent,shiftp_ent ])
    regressors_name = regressors_name + ['syl_' + name for name in phonemic_data['names']] + ['syl_lowtemp_sem', 'syl_hightemp_sem', 'syl_scandal', 'syl_entropy t-1', 'syl_entropy t-2', 'syl_entropy t+1']
    
    return regressors, regressors_name