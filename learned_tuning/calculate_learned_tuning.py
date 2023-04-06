import numpy as np
from learned_tuning.Bayesian_decoder import Bayesian_decoder

def cal_unit_learned_tuning(posterior_prob, unit_spike_counts_concat):
    
    # spike-weighted average of posteriors divided by average of the
    # posteriors

    n_time_bins = len(unit_spike_counts_concat)
    n_pos_bins = posterior_prob.shape[0]

    weighted_summation = np.sum(np.tile(unit_spike_counts_concat, (n_pos_bins, 1)) * posterior_prob, axis=1) / n_time_bins
    p_of_x = np.sum(posterior_prob, axis=1) / n_time_bins

    learned_tuning = weighted_summation / p_of_x

    return learned_tuning


    
def calculate_learned_tuning(PBEs, spikes, L_ratios, time_bin_duration):

    """
    Calculates learned tuning curves for each unit based on its spiking activity during population burst events (PBEs),
    as well as the spatial tuning curves of other coactive units, using Bayesian decoding.

    Parameters:
    -----------
    PBEs: list
        A list of dictionaries containing information including the spike counts in each time bins for each unit .
    spikes: list
        A list of dictionaries containing spike information inclduing the place fields for each unit
    L_ratios: list
        A list of dictionaries containing L ratio between pairs of units
    time_bin_duration: float
        The duration of each time bin in seconds.

    Returns:
    --------
    learned_tunings: array
        An array of learned tuning curves, where each row corresponds to a unit and each column corresponds to a spatial bin.
    """

    
    # make sure the units are matched between spikes and PBEInfo.fr_20msbin
    if len(spikes) != PBEs[0]['fr_20msbin'].shape[1]:
        raise ValueError('There is a mismatch in number of units between spikes and PBEInfo.fr_20msbin')
    
    
    num_units = len(spikes)

    shank_ids = np.zeros((num_units, 1))
    cluster_ids = np.zeros((num_units, 1))

    for unit in range(num_units):
        shank_ids[unit]   = spikes[unit]['shank_id']-1 # because the shank indexing for in the L-ratios array starts at 0
        cluster_ids[unit] = spikes[unit]['cluster_id']
    
    # place fields of each unit needed for calculating the posteriors 
    if 'RL' in spikes[0]['place_fields']:
        two_place_fields_flag = 1
        num_pos_bins = len(spikes[0]['place_fields']['RL'])
        
        place_fields_RL = np.zeros((num_units, num_pos_bins))
        place_fields_LR = np.zeros((num_units, num_pos_bins))
        for unit in range(num_units):
            place_fields_LR[unit, :] = spikes[unit]['place_fields']['LR']
            place_fields_RL[unit, :] = spikes[unit]['place_fields']['RL']
            
        place_fields_LR[place_fields_LR == 0] = 1e-4
        place_fields_RL[place_fields_RL == 0] = 1e-4
    else:
        two_place_fields_flag = 0
        num_pos_bins = len(spikes[0]['place_fields']['uni'])
        
        place_fields = np.zeros((num_units, num_pos_bins))
        for unit in range(num_units):
            place_fields[unit, :] = spikes[unit]['spatialTuning_smoothed']['uni']
            
        place_fields[place_fields == 0] = 1e-4
    
    # calculate Lratio threshold by pooling across all shanks
    # Lratio = []
    # for ii = 1:numel(L_ratios)
    #     temp = L_ratios(ii).Lratio .*(1-np.eye(L_ratios(ii).Lratio.shape[0])) # excluding the diagonal
    #     Lratio = np.concatenate((Lratio, temp.flatten()))
    # Lratio_thresh = np.median(Lratio)

    L_ratio_thresh = 1e-3
    

    num_PBEs = len(PBEs)
    
    PBE_each_bin_spike_counts = []
    for pbe in range(num_PBEs):
        PBE_each_bin_spike_counts.append(PBEs[pbe]['fr_20msbin'].transpose())
        
    # concatenate the spike counts per time bin across all PBEs 
    PBE_spike_counts_concat = np.concatenate(PBE_each_bin_spike_counts, axis=1)


    active_units = np.where(np.sum(PBE_spike_counts_concat, axis=1) > 0)[0] # detect units that fired at least once during the PBEs
    num_active_units = len(active_units)

    # num_coactive_units_with_each_unit = np.zeros((num_units, 1))
    learned_tunings = np.zeros((num_units, num_pos_bins))

    for unit in range(num_active_units):
        
        curr_unit = active_units[unit]
        
        curr_unit_cluster_id = cluster_ids[curr_unit]
        curr_unit_shank_id = shank_ids[curr_unit]
        
        curr_unit_cluster_id = curr_unit_cluster_id.astype(int)[0]
        curr_unit_shank_id   = curr_unit_shank_id.astype(int)[0]

        included_units_from_other_shanks = np.where(shank_ids[:, 0] != curr_unit_shank_id)[0]
        
        idx = np.where(L_ratios[curr_unit_shank_id]['cluster_ids'] == curr_unit_cluster_id)[0]
        less_than_L_ratio_thresh = L_ratios[curr_unit_shank_id]['L_ratios'][:, idx] < L_ratio_thresh
        accepted_clusters = L_ratios[curr_unit_shank_id]['cluster_ids'][less_than_L_ratio_thresh]
        
        included_units_from_same_shank = np.where((shank_ids == curr_unit_shank_id) & np.isin(cluster_ids, accepted_clusters))[0]
        
        other_units = np.concatenate((included_units_from_other_shanks, included_units_from_same_shank))
        other_units = np.sort(other_units)
        

        unit_PBE_spike_counts_concat = PBE_spike_counts_concat[curr_unit, :]
        other_units_PBE_spike_counts_concat = PBE_spike_counts_concat[other_units, :]
        
        # num_coactive_units_with_each_unit[curr_unit] = np.sum(other_units_PBE_spike_counts_concat > 0, axis=0)
        
        if two_place_fields_flag == 1:

            posterior_prob_RL = Bayesian_decoder(other_units_PBE_spike_counts_concat, place_fields_RL[other_units, :], time_bin_duration)
            posterior_prob_LR = Bayesian_decoder(other_units_PBE_spike_counts_concat, place_fields_LR[other_units, :], time_bin_duration)
            
            posterior_prob = posterior_prob_RL + posterior_prob_LR
            posterior_prob = posterior_prob/np.tile(np.sum(posterior_prob, axis=0), (num_pos_bins, 1))
            
            learned_tunings[curr_unit, :] = cal_unit_learned_tuning(posterior_prob, unit_PBE_spike_counts_concat)
            
        else:
            
            posterior_prob = Bayesian_decoder(other_units_PBE_spike_counts_concat, place_fields[other_units, :], time_bin_duration)
            posterior_prob = posterior_prob/np.tile(np.sum(posterior_prob, axis=0), (num_pos_bins, 1))
            
            learned_tunings[curr_unit, :] = cal_unit_learned_tuning(posterior_prob, unit_PBE_spike_counts_concat)

    return learned_tunings