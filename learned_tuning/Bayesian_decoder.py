import numpy as np

def calculate_posterior_probability(spike_counts, log_tunings, tuning_summation):

    num_position_bins = log_tunings.shape[1] # log_tunings has the same shape as place_fields
    num_time_bins = spike_counts.shape[1]

    # Calculate posterior probability for a given set of spike counts
    log_pr = np.sum(np.tile(np.transpose(log_tunings[:,:,np.newaxis], (1,2,0)), [1, num_time_bins, 1]) *
                    np.tile(np.transpose(spike_counts[:,:, np.newaxis], (2,1,0)), [num_position_bins, 1, 1]), axis=2)
    
    log_pr += np.tile(np.transpose(tuning_summation[np.newaxis,:], (1, 0)), [1, num_time_bins])
    
    post_pr = np.exp(log_pr)

    return post_pr

def Bayesian_decoder(spike_count_by_time_bin, place_fields, time_bin_duration):
    
    # This function takes in three parameters:
    # - spike_count_by_time_bin: a 2D numpy array with shape (num_neurons, num_num_time_bins) that contains the number of spikes
    #   observed by each neuron in each time bin
    # - place_fields: a 2D numpy array with shape (num_neurons, num_pos_bins) that contains the estimated firing rates of each neuron
    #   in each spatial bin
    # - time_bin_duration: the duration of each time bin in seconds

    # The function uses a Bayesian decoding algorithm to estimate the posterior probability distribution over spatial bins,
    # given the observed spike counts. Specifically, it calculates the log probabilities (logPr) using the formula:

    # logPr = -tau * (sum_{i=1}^N lambda_i) + sum_{i=1}^N n_i * log(tau * lambda_i)

    # where tau is the time bin duration, N is the number of neurons, lambda_i is the estimated firing rate of neuron i in each
    # spatial bin, and n_i is the observed spike count of neuron i in each time bin.

    # The log probabilities are then converted back to probability values by calculating the exponentials of the logPr values:

    # prob = exp(logPr)

    # The function returns a 2D numpy array with shape (num_pos_bins, num_num_time_bins) that contains the posterior probability
    # distribution over spatial bins for each time bin.

    # pdb.set_trace()

    num_position_bins = place_fields.shape[1] 
    num_time_bins = spike_count_by_time_bin.shape[1]
    
    # Doing calculations that are independent of the spike counts within each time bin
    log_tunings = np.log(time_bin_duration * place_fields)
    tuning_summation = - time_bin_duration * np.sum(place_fields, axis=0)

    max_num_bin_in_part = 1000    

    if num_time_bins < max_num_bin_in_part:

        # This is suitable if we are decoding bins within individual PBEs
        post_pr = calculate_posterior_probability(spike_count_by_time_bin, log_tunings, tuning_summation)
        
    else:
        # To decrease memory load when the number of time bins are high, we divide the data into smaller parts,
        #  calculated posterior and at the end concatenate all parts 
 
        n_parts = np.ceil(num_time_bins/max_num_bin_in_part).astype(int)
        
        post_pr = np.empty((num_position_bins, num_time_bins))
        
        for ii in range(n_parts):
            curr_time_bins  = np.arange((ii*max_num_bin_in_part), min((ii+1)*max_num_bin_in_part, num_time_bins))
            
            curr_post_pr = calculate_posterior_probability(spike_count_by_time_bin[:, curr_time_bins], log_tunings, tuning_summation)

            post_pr[:, curr_time_bins] = curr_post_pr
    
    return post_pr