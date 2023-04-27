import numpy as np
from learned_tuning.Bayesian_decoder import Bayesian_decoder

def cal_unit_learned_tuning(posterior_prob, unit_spike_counts_concat):
    
    # spike-weighted average of posteriors divided by average of the
    # posteriors

    n_time_bins = len(unit_spike_counts_concat)
    n_pos_bins = posterior_prob.shape[0]

    weighted_summation = np.sum(np.tile(unit_spike_counts_concat, (n_pos_bins, 1)) * posterior_prob, axis=1) / n_time_bins
    p_of_x = np.sum(posterior_prob, axis=1) / n_time_bins

    unit_learned_tuning = weighted_summation / p_of_x

    return unit_learned_tuning


    
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
            place_fields[unit, :] = spikes[unit]['place_fields']['uni']
            
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
        
        # other_units = list(set(range(num_active_units)) - set([curr_unit]))
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

def cal_unit_learned_tuning_with_p_of_x(posterior_prob, unit_spike_counts_concat, p_of_x):
    
    # spike-weighted average of posteriors divided by average of the
    # posteriors

    n_time_bins = len(unit_spike_counts_concat)
    n_pos_bins = posterior_prob.shape[0]

    weighted_summation = np.sum(np.tile(unit_spike_counts_concat, (n_pos_bins, 1)) * posterior_prob, axis=1) / n_time_bins

    unit_learned_tuning = weighted_summation / p_of_x

    return unit_learned_tuning


def calculate_learned_tuning_PBE_subsets(PBEs, spikes, PBE_subset_indices, L_ratios, time_bin_duration):
    """
    Calculates learned tuning curves for each unit based on its spiking activity during different subsets of population burst events (PBEs),
    as well as the spatial tuning curves of other coactive units, using Bayesian decoding.

    Parameters:
    -----------
    PBEs: list
        A list of dictionaries containing information including the spike counts in each time bins for each unit within each PBE.
    spikes: list
        A list of dictionaries containing spike information inclduing the place fields for each unit
    PBE_subset_indices: list
        A list containing indices of PBEs in various subsets used to calculate learned tunings 
    L_ratios: list 
        A list of dictionaries containing L ratio between pairs of units on the same shank
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
            place_fields[unit, :] = spikes[unit]['place_fields']['uni']
            
        place_fields[place_fields == 0] = 1e-4

    L_ratio_thresh = 1e-3


    total_num_PBEs = len(PBEs)
    
    PBE_each_bin_spike_counts = []
    for pbe in range(total_num_PBEs):
        PBE_each_bin_spike_counts.append(PBEs[pbe]['fr_20msbin'].transpose())
        
    # concatenate the spike counts per time bin across all PBEs 
    PBE_spike_counts_concat = np.concatenate(PBE_each_bin_spike_counts, axis=1)

    active_units = np.where(np.sum(PBE_spike_counts_concat, axis=1) > 0)[0] # detect units that fired at least once during the PBEs
    num_active_units = len(active_units)

    del PBE_spike_counts_concat

    num_PBE_subset = len(PBE_subset_indices)
    learned_tunings_PBE_subsets = np.full((num_units, num_pos_bins, num_PBE_subset), np.nan)

    num_dots = int(num_active_units * (10/100)) # for printing how far we are in the calculations, what tenth of units have been processed 
    count = 0

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
        
        # other_units = list(set(range(num_active_units)) - set([curr_unit]))
        other_units = np.sort(other_units)
        
        unit_PBE_spike_counts, other_units_PBE_spike_counts, posterior_prob = [[None for _ in range(total_num_PBEs)] for _ in range(3)]
        
        for pbe in range(total_num_PBEs):

            unit_PBE_spike_counts[pbe] = PBE_each_bin_spike_counts[pbe][curr_unit]
            other_units_PBE_spike_counts = PBE_each_bin_spike_counts[pbe][other_units]

            if two_place_fields_flag == 1:

                posterior_prob_RL = Bayesian_decoder(other_units_PBE_spike_counts, place_fields_RL[other_units, :], time_bin_duration)
                posterior_prob_LR = Bayesian_decoder(other_units_PBE_spike_counts, place_fields_LR[other_units, :], time_bin_duration)
                
                posterior_prob[pbe] = posterior_prob_RL + posterior_prob_LR
                posterior_prob[pbe] = posterior_prob[pbe]/np.tile(np.sum(posterior_prob[pbe], axis=0), (num_pos_bins, 1))
                
            else:
                
                posterior_prob[pbe] = Bayesian_decoder(other_units_PBE_spike_counts, place_fields[other_units, :], time_bin_duration)
                posterior_prob[pbe] = posterior_prob[pbe]/np.tile(np.sum(posterior_prob[pbe], axis=0), (num_pos_bins, 1))

                

        # Calculate separate learned tunings using PBEs within each PBE subset

        posterior_prob_concatenated_all_PBEs = np.concatenate(posterior_prob, axis=1)
        posterior_prob_concatenated_all_PBEs = posterior_prob_concatenated_all_PBEs[:, ~np.isnan(np.sum(posterior_prob_concatenated_all_PBEs, axis=0))]

        p_of_x = np.mean(posterior_prob_concatenated_all_PBEs, axis=1)
        del posterior_prob_concatenated_all_PBEs

        for pbe_sub in range(num_PBE_subset):

            curr_PBEs = PBE_subset_indices[pbe_sub].astype(int)

            if len(curr_PBEs) > 0:

                posterior_prob_sub = np.concatenate([posterior_prob[i] for i in curr_PBEs], axis=1)
                idx = ~np.isnan(np.sum(posterior_prob_sub, axis = 0))

                unit_PBE_spike_counts_concatenated_sub = np.concatenate([unit_PBE_spike_counts[i] for i in curr_PBEs])

                learned_tunings_PBE_subsets[curr_unit, :, pbe_sub] = cal_unit_learned_tuning_with_p_of_x(posterior_prob_sub, unit_PBE_spike_counts_concatenated_sub, p_of_x)

        if (unit+1) % num_dots == 1:
                count += 1
                message = "." * count
                print(message, end="\r")

    return learned_tunings_PBE_subsets



def calculate_all_column_correlations(matrix1, matrix2):
    """
    Calculates the correlation between each column of the first matrix with every other column of the second matrix,
    and stores the correlation coefficients in an output matrix.

    Args:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.

    Returns:
        np.ndarray: A 2-dimensional array containing the correlation coefficients between each column of the first matrix
                    and every other column of the second matrix. The shape of the output matrix is (n, m), where n is the
                    number of columns in the first matrix, and m is the number of columns in the second matrix.
    """
    # Check if the input matrices have the same number of columns
    if matrix1.shape[1] != matrix2.shape[1]:
        raise ValueError("Input matrices must have the same number of columns")

    # Initialize an empty matrix to store the correlation coefficients
    correlations = np.empty((matrix1.shape[1], matrix2.shape[1]))

    # Loop through each column of the first matrix
    for i in range(matrix1.shape[1]):
        # Select the column of interest from the first matrix
        column1 = matrix1[:, i]

        # Loop through each column of the second matrix and calculate the correlation
        for j in range(matrix2.shape[1]):
            corr = np.corrcoef(column1, matrix2[:, j])[0, 1]
            correlations[i, j] = corr

    return correlations

import numpy as np


def random_column_sampling(matrix):
    """
    Randomly samples a column for each row of a given matrix.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: A 1-dimensional array containing the randomly sampled values from the columns of the input matrix.
                    The length of the output array is equal to the number of rows in the input matrix.
    """
    # Get the number of rows in the input matrix
    num_rows = matrix.shape[0]

    # Generate random column indices for each row
    column_indices = np.random.randint(0, matrix.shape[1], num_rows)

    # Extract the values at the randomly sampled column indices for each row
    sampled_values = matrix[np.arange(num_rows), column_indices]

    return sampled_values



def calculate_place_field_fidelity_of_learned_tuning(learned_tunings, place_fields, num_shuffles):

    """"
    calcualtes the place field fidelty of the learned tuning 

    Args:
    learned_tunings (np.ndarray): Learned tunings (num_units * num_pos_bins).
    place_fields (np.ndarray): Place fields (num_units * num_pos_bins).
    num_shuffles (int): Number of datasets and corresponding distribution of null distribution by shuffling the unit identities of the place fields when calculating the place field fidelities.

Returns:
    learned_tuning_place_field_pearson_corr (np.ndarray): 1-dimensional array of Pearson correlation coefficients between the learned tunings and the place fields (num_units,).
    median_LT_PF_pearson_corr (dict): A dictionary with the following keys and values:
        - "data": a scalar value representing the median Pearson correlation coefficient between the learned tunings and the place fields.
        - "PF_unit_IDX_shuffle": a 1-dimensional array containing num_shuffles Pearson correlation coefficients between the shuffled place fields and the learned tunings.
        - "p_value": a scalar value representing the p-value of the median Pearson correlation coefficient.
    """
    # calculate the Pearson correlation between the learned tunings and place fields

    learned_tuning_place_field_pearson_corr_all_combinations = calculate_all_column_correlations(np.transpose(learned_tunings), np.transpose(place_fields))    

    # Pearson correlation between learned tunings and place fields of identical units
    learned_tuning_place_field_pearson_corr = np.diag(learned_tuning_place_field_pearson_corr_all_combinations)


   # statistical significance of the median of the distribution
   #  
    median_LT_PF_pearson_corr = dict()

    median_LT_PF_pearson_corr["data"] = np.nanmedian(learned_tuning_place_field_pearson_corr)

    # A null hypothesis is that the learend tuning for a unit is as correalted with its correposnding place fields 
    # as it is with the place fiels of a randomly drawn unit. To test against this hypothesis we match each learned tuning
    # with a place field of a random unit and calaculte the correlation between them

    median_LT_PF_pearson_corr["PF_unit_IDX_shuffle"] = np.empty((num_shuffles,))

    for i_shuffle in range(num_shuffles):
        random_data = random_column_sampling(learned_tuning_place_field_pearson_corr_all_combinations)
        median_LT_PF_pearson_corr["PF_unit_IDX_shuffle"][i_shuffle] = np.nanmedian(random_data)

    median_LT_PF_pearson_corr["p_value"] = len(np.where(median_LT_PF_pearson_corr["PF_unit_IDX_shuffle"] >= median_LT_PF_pearson_corr["data"])[0]) / num_shuffles

    return learned_tuning_place_field_pearson_corr, median_LT_PF_pearson_corr 

