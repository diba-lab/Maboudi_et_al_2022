function [learnedTunings, px] = calLearnedTunings(PBEInfo, spikes, selectPBEs, clusterQuality)

% This function calculates the learned tunings based on the PBEs


% PBEInfo: A data structre for all pieces of information for individual
% PBEs including their timing, epoch, ripple amplitude, MU amplitude,
% binned firing rates, etc. 

% spikes: the structure containing the spike times, unit identities, place fields, etc

% selectPBEs: is used if we want to calculate the learned tunings based on
% a limited set of PBEs

% cluster quality: a data strcuture used to limite the other units to a
% set with minimum overlap in spike features with a given unit



binDur = 0.01; % duration of each PBE time bins needed for calculating the posterior probabilities

% make sure the units are matched between spikes and PBEInfo.fr_20msbin
if numel(spikes) ~= size(PBEInfo(1).fr_20msbin)
    error('There is a mismatch in the number of units between spikes and PBEInfo.fr_20msbin')
end

nUnits = numel(spikes);
unitIDs = nan(nUnits, 2);
for iUnit = 1:nUnits
    unitIDs(iUnit, :) = spikes(iUnit).id;
end


% place fields of each unit needed for calculating the posteriors 

if isfield(spikes(1).spatialTuning_smoothed, 'RL')
    
    twoPlaceFieldsFlag = 1;
    nPosBins = numel(spikes(1).spatialTuning_smoothed.RL);
    
    placeFieldsLR = zeros(nUnits, nPosBins);
    placeFieldsRL = zeros(nUnits, nPosBins);    
    for iUnit = 1:nUnits
        placeFieldsRL(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.RL;
        placeFieldsLR(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.LR;
    end
    
    placeFieldsRL(~placeFieldsRL) = 1e-4;
    placeFieldsLR(~placeFieldsLR) = 1e-4;
    
else
    twoPlaceFieldsFlag = 0;
    nPosBins = numel(spikes(1).spatialTuning_smoothed.uni);
    
    placeFields = zeros(nUnits, nPosBins);
    for iUnit = 1:nUnits
        placeFields(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.uni;
    end
    
    placeFields(~placeFields) = 1e-4;
end
    


% Lratio threshold 
Lratio_thresh = 1e-3;


% what subset of PBEs are going to be included in the analysis (used in particular for calculating learned tunings from PBEs with replay scores in a specific range)
if isempty(selectPBEs) 
    selectPBEs = 1:numel(PBEInfo);
end

nPBEs = numel(selectPBEs);
PBEbinnedFirings = cell(nPBEs, 1);
for pbe = 1:numel(selectPBEs)
    PBEbinnedFirings{pbe} = PBEInfo(selectPBEs(pbe)).fr_20msbin; 
end



% concatenate the PBEs 
concatPBEfirings = cell2mat(PBEbinnedFirings');

activeUnits  = find(sum(concatPBEfirings, 2) > 0); % considering only units that fired at least once during the PBEs
nActiveUnits = numel(activeUnits);


learnedTunings  = zeros(nUnits, nPosBins);
px              = zeros(nUnits, nPosBins);


for iUnit = 1:nActiveUnits

    
    currUnit = activeUnits(iUnit);
    
    currUnitClu   = unitIDs(currUnit, 2);
    currUnitShank = unitIDs(currUnit, 1);
    
    
    % all units from the other shanks are included
    includedUnits_otherShanks = find(unitIDs(:, 1) ~= currUnitShank); 
      
    
    % among units on the same shank as of the current unit, units with L-ratio less than threshold L-ratio are included 
    
    idx = clusterQuality(currUnitShank).clus == currUnitClu; % row index corresponding to the current unit
    isLessThanThreshLratio = clusterQuality(currUnitShank).Lratio(:, idx) < Lratio_thresh; % idx of the columns with L-ratio below threshold
    acceptedClusters = clusterQuality(currUnitShank).clus(isLessThanThreshLratio); % the clusters corresponding to the qualified columns
    
    includedUnits_sameShanks = find(unitIDs(:,1) == currUnitShank & ismember(unitIDs(:,2), acceptedClusters));
    
    
    
    otherUnits = [includedUnits_otherShanks; includedUnits_sameShanks];
    otherUnits = sort(otherUnits, 'ascend');


    concatPBEfirings_unit       = concatPBEfirings(currUnit, :);   
    concatPBEfirings_OtherUnits = concatPBEfirings(otherUnits, :); 

    
    % posterior probabilities given each direction

    if twoPlaceFieldsFlag == 1
        
        postprobRL = baysDecoder(concatPBEfirings_OtherUnits, placeFieldsRL(otherUnits, :), binDur);  
        postprobLR = baysDecoder(concatPBEfirings_OtherUnits, placeFieldsLR(otherUnits, :), binDur);  


        %%% marginalized over directions %%%
        posteriorProbMatrix = postprobRL + postprobLR;
        posteriorProbMatrix = posteriorProbMatrix./repmat(sum(posteriorProbMatrix, 1), [nPosBins, 1]); 

        [learnedTunings(currUnit, :), px(currUnit, :)] = calLT(posteriorProbMatrix, concatPBEfirings_unit);
      
    else

        posteriorProbMatrix = baysDecoder(concatPBEfirings_OtherUnits, placeFields(otherUnits, :), binDur); 
        posteriorProbMatrix = posteriorProbMatrix./repmat(sum(posteriorProbMatrix, 1), [nPosBins, 1]);

        [learnedTunings(currUnit, :), px(currUnit, :)] = calLT(posteriorProbMatrix, concatPBEfirings_unit);


    end

end

end


function [assemblyTunings, p_of_x] = calLT(posteriorProbMatrix, unitFirings)
    
    % spike-weighted average of posteriors divided by average of the
    % posteriors
    
    nTimeBins = numel(unitFirings);
    nPosBins  = size(posteriorProbMatrix, 1);


    weightedSummation = sum(repmat(unitFirings, [nPosBins 1]) .* posteriorProbMatrix, 2)/nTimeBins;
    p_of_x = sum(posteriorProbMatrix, 2)/nTimeBins;
    
    
    assemblyTunings = weightedSummation ./ p_of_x; 
    
end

