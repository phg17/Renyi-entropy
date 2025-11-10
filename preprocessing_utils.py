import numpy as np
import scipy.signal as sig
import re
from text_utils import get_alphabet, remove_duplicates
from string import digits
from sklearn.decomposition import FastICA
import mat73

def mono_to_bipolar(data_subject, channel_subject, locations = False):
    '''
    Converts monopolar data into bipolar data
    input : data_subject : np.array(N_times, N_channels) : EEG data
            channel_subject : list(N_channels) : list of channel names
            locations : np.array(3, N_channels) : position of electrodes in space
    '''
    pattern = r'[0-9]'
    data_bipolar = {
        subject: np.array([data_subject[subject][:, i+1] - data_subject[subject][:, i] 
                           for i in range(len(channel_subject[subject])-1)
                           if re.sub(pattern, '', channel_subject[subject][i]) == re.sub(pattern, '', channel_subject[subject][i+1])]).T
        for subject in data_subject
    }
    channel_bipolar = {
        subject: [f"{channel_subject[subject][i+1]}-{channel_subject[subject][i]}"
                  for i in range(len(channel_subject[subject])-1)
                  if re.sub(pattern, '', channel_subject[subject][i]) == re.sub(pattern, '', channel_subject[subject][i+1])]
        for subject in data_subject
    }
    if locations:
        locations_bipolar = {
            subject: np.array([(locations[subject][:,i+1] + locations[subject][:,i])/2
                      for i in range(len(channel_subject[subject])-1)
                      if re.sub(pattern, '', channel_subject[subject][i]) == re.sub(pattern, '', channel_subject[subject][i+1])]).T
            for subject in data_subject
        }
        return data_bipolar, channel_bipolar, locations_bipolar
    else:
        return data_bipolar, channel_bipolar

def select_channels(eeg, channels, locations = None, channel_select = [], strict = True, strict_shaft = False, exclude = False):
    '''
    Selects the channels from the EEG data which names contains one of the strings in channel_select
    input : eeg : np.array : EEG data
            channels : list : list of channel names
            channel_select : list : list of strings to search in the channel names
            strict : bool : if True, the channel name must only contain the strings in channel_select
                            for example, if strict == True: HT is not valid. H' will be however, this only
                            applies to alphabetical characters
            strict_shaft : bool : if True, the channel name will also take into account the difference between H and H'
            exclude : bool : if True, channels with the strings will instead be deleted
    '''
    new_eeg = []
    new_channels = []
    new_locations = []
    location_flag = True
    if locations is None:
        location_flag = False
        locations = np.zeros((3,eeg.shape[1]))
    for index_channel in range(eeg.shape[1]):
        channel = channels[index_channel]
        channel_strict = remove_duplicates(get_alphabet(channel))
        channel_strict_shaft = remove_duplicates(get_alphabet(channel, regex_val = "[^a-zA-Z']"))
        if not strict and not exclude:
            if any([channel_select[i] in channel for i in range(len(channel_select))]):
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
        elif strict_shaft and not exclude:
            if any([channel_select[i] == channel_strict_shaft for i in range(len(channel_select))]):
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
        elif strict and not exclude:
            if any([channel_select[i] == channel_strict for i in range(len(channel_select))]):
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
        else:
            if any([channel_select[i] in channel for i in range(len(channel_select))]):
                True
            else:
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
    if location_flag:
        return np.asarray(new_eeg).T, new_channels, np.asarray(new_locations).T
    else: 
        return np.asarray(new_eeg).T, new_channels


def pick_channels(eeg, channels, locations = None, channel_select = [],exclude = False):
    '''
    Selects/Rejects the channels from the EEG data which names are exacly the ones given in channel_select
    input : eeg : np.array : EEG data
            channels : list : list of channel names
            channel_select : list : list of strings to search in the channel names
            exclude : bool : if True, channels with the strings will instead be deleted
    '''

    new_eeg = []
    new_channels = []
    new_locations = []
    location_flag = True
    if locations is None:
        location_flag = False
        locations = np.zeros((3,eeg.shape[1]))
    for index_channel in range(eeg.shape[1]):
        channel = channels[index_channel]
        if not exclude:
            if channel  in channel_select:
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
        else:
            if not channel in channel_select:
                new_eeg.append(eeg[:,index_channel])
                new_channels.append(channel)
                new_locations.append(locations[:,index_channel])
    if location_flag:
        return np.asarray(new_eeg).T, new_channels, np.asarray(new_locations).T
    else: 
        return np.asarray(new_eeg).T, new_channels


def available_regions(channels):
    '''
    Returns the available regions in the EEG data
    input : channels : list : list of channel names
    '''
    regions = []
    for channel in channels:
        region = re.sub(r'[0-9]', '', channel)
        if region not in regions:
            regions.append(region)
    return regions

def delete_channels(data, channels, delete_list):
    '''
    Delete the channels from the list
    '''
    new_data = []
    new_channels = []
    for eeg, chan in zip(data.T, channels):
        if not any([chan == delete_chan for delete_chan in delete_list]):
            new_data.append(eeg)
            new_channels.append(chan)
    return np.asarray(new_data).T, new_channels

def adj_scale(signal,min = 0, max = 1):
    #Scales a signal to a range
    return (signal - np.min(signal)) * (max - min) / (np.max(signal) - np.min(signal)) + min


def ica_shaft(eeg, channels,locations = None, return_mixing =  False, random_state = None):
    remove_digits = str.maketrans('', '', digits)
    shafts = available_regions(channels)
    eeg_ica = np.empty((eeg.shape[0],0))
    mixing_ica = []
    channels_ica = []
    if locations is None:
        for shaft in shafts:
            eeg_shaft, channels_shaft = select_channels(eeg,channels, channel_select = [shaft], exclude = False, strict = True, strict_shaft=True)
            if len(eeg_shaft.shape) == 2:
                ica = FastICA(n_components=eeg_shaft.shape[1], random_state = random_state)
                eeg_shaft_ica = ica.fit_transform(eeg_shaft)
                mixing_ica += [mixing_matrix for mixing_matrix in ica.mixing_]
                channels_ica += [shaft + '_ICA_' + str(component_index) for component_index in range(eeg_shaft.shape[1])]
                eeg_ica = np.hstack([eeg_ica, eeg_shaft_ica])
                
        if return_mixing:
            return eeg_ica, channels_ica, mixing_ica
        else:
            return eeg_ica, channels_ica, 
    else:
        locations_ica = np.empty((locations.shape[0],0))
        for shaft in shafts:
            eeg_shaft, channels_shaft, locations_shaft = select_channels(eeg,channels, locations, channel_select = [shaft], exclude = False, strict = True, strict_shaft=True)
            if len(eeg_shaft.shape) == 2:
                ica = FastICA(n_components=eeg_shaft.shape[1], random_state = random_state)
                eeg_shaft_ica = ica.fit_transform(eeg_shaft)
                mixing_ica += [mixing_matrix for mixing_matrix in ica.mixing_]
                channels_ica += [shaft + '_ICA_' + str(component_index) for component_index in range(eeg_shaft.shape[1])]
                locations_shaft_ica = ica.transform(locations_shaft)
                eeg_ica = np.hstack([eeg_ica, eeg_shaft_ica])
                locations_ica = np.hstack([locations_ica, locations_shaft_ica])
        if return_mixing:
            return eeg_ica, channels_ica, locations_ica, mixing_ica
        else:
            return eeg_ica, channels_ica, locations_ica

def get_bipolar_atlas(path_bipolar_atlas, atlas = 'Destrieux', n_labels = 1):
    bipolar_atlas = dict()
    data_bi = mat73.loadmat(path_bipolar_atlas)
    assert atlas in data_bi, "atlas must be one of" + str(list(data_bi.keys()))
    for electrode_index, electrode in enumerate(data_bi['coi']['label']):
        bipolar_atlas[electrode[0].split("_")[0] + electrode[0].split("_")[2] + "-" + electrode[0].split("_")[0] + electrode[0].split("_")[1]] = data_bi[atlas]['label'][electrode_index][:n_labels]
    
    return bipolar_atlas
