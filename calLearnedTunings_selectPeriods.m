function [learnedTunings, concatBinCenters, concatBoutIDs, boutFirings_unit] = calLearnedTunings_selectPeriods(bouts, spikes, binDur, clusterQuality)




% binDur = 0.02; % duration of the time bins used for calculating the posterior probabilities

nUnits = numel(spikes);

unitIDs = zeros(nUnits, 2);
for iUnit = 1:nUnits
    unitIDs(iUnit, :) = spikes(iUnit).id;
end




% place fields of each unit needed for calculating the posteriors 

if isfield(spikes(1).spatialTuning_smoothed, 'RL')
    
    for ii = 1:numel(spikes)

        spikes(ii).spatialTuning_smoothed.RL = spikes(ii).spatialTuning_smoothed.RL;
        spikes(ii).spatialTuning_smoothed.LR = spikes(ii).spatialTuning_smoothed.LR;
    end
    
    twoPlaceFieldsFlag = 1;
    nPosBins = numel(spikes(1).spatialTuning_smoothed.RL);
    
    placeFieldsLR = zeros(nUnits, nPosBins);
    placeFieldsRL = zeros(nUnits, nPosBins);    
    placeFields   = zeros(nUnits, nPosBins); % spatial tunings without considering running directions
    
    for iUnit = 1:nUnits
        placeFieldsRL(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.RL;
        placeFieldsLR(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.LR;
        placeFields(iUnit, :)   = spikes(iUnit).spatialTuning_smoothed.uni;
    end
    
    placeFieldsRL(~placeFieldsRL) = 1e-4;
    placeFieldsLR(~placeFieldsLR) = 1e-4;
    placeFields(~placeFields)     = 1e-4;

else

    twoPlaceFieldsFlag = 0;
    nPosBins = numel(spikes(1).spatialTuning_smoothed.uni);
    
    placeFields = zeros(nUnits, nPosBins);
    for iUnit = 1:nUnits
        placeFields(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.uni;
    end
    
    placeFields(~placeFields) = 1e-4;
end
    
Lratio_thresh = 1e-3; % the threshold on L-ratio


% Divide the whole period to 20 ms time bins

fileInfo.okUnits = 1:nUnits;

nBouts = size(bouts, 1);

binnedFirings = cell(nBouts, 1);
binCenters    = cell(nBouts, 1);
boutID        = cell(nBouts, 1);
for ib = 1:nBouts
    [binnedFirings{ib}, binCenters{ib}] = timeBinning_20(bouts(ib, :), spikes, binDur, 0, fileInfo);
    
    nBins = size(binnedFirings{ib}, 2);

    boutID{ib} = ib*ones(nBins, 1);
end

concatBinCenters    = cell2mat(binCenters');
concatBoutIDs       = cell2mat(boutID);
concatBinnedFirings = cell2mat(binnedFirings');
ntTimeBins          = numel(concatBinCenters);


activeUnits  = find(sum(concatBinnedFirings, 2) > 0);
nActiveUnits = numel(activeUnits); 


learnedTunings = nan(nUnits, nPosBins);


boutFirings_unit  = nan(nUnits, ntTimeBins); 

for iUnit = 1:nActiveUnits
    
    currUnit = activeUnits(iUnit);
    
    currUnitClu   = unitIDs(currUnit, 2);
    currUnitShank = unitIDs(currUnit, 1);
    
    
    % all units from the other shanks are included
    includedUnits_otherShanks = find(unitIDs(:, 1) ~= currUnitShank); 
    
    
    
    % among units on the same shank as of the current unit, units with L-ratio less than threshold L-ratio are included 
    
    idx = clusterQuality(currUnitShank).clus == currUnitClu; % row index corresponding to the current unit
    isLessThanMaxLratio = clusterQuality(currUnitShank).Lratio(:, idx) < Lratio_thresh; % idx of the columns with L-ratio below threshold
    acceptedClusters = clusterQuality(currUnitShank).clus(isLessThanMaxLratio); % the clusters corresponding to the qualified columns
    
    includedUnits_sameShanks = find(unitIDs(:,1) == currUnitShank & ismember(unitIDs(:,2), acceptedClusters));
    
    
    otherUnits = [includedUnits_otherShanks; includedUnits_sameShanks];
    otherUnits = sort(otherUnits, 'ascend');

    
    boutFirings_unit_bout  = cell(nBouts, 1);
    boutFirings_otherUnits = cell(nBouts, 1);
    posteriorProbMatrix    = cell(nBouts, 1);

    for ib = 1:nBouts % doing the calculations separately for each boat, however we could have concatenated and avoided the for loop
        
        boutFirings_unit_bout{ib}  = binnedFirings{ib}(currUnit, :);
        boutFirings_otherUnits{ib} = binnedFirings{ib}(otherUnits, :);


        if twoPlaceFieldsFlag == 1

            postprobRL = baysDecoder(boutFirings_otherUnits{ib}, placeFieldsRL(otherUnits, :), binDur);  
            postprobLR = baysDecoder(boutFirings_otherUnits{ib}, placeFieldsLR(otherUnits, :), binDur);  

            posteriorProbMatrix{ib} = postprobRL + postprobLR;
 
        else
            posteriorProbMatrix{ib} = baysDecoder(boutFirings_otherUnits{ib}, placeFields(otherUnits, :), binDur); 
            
        end
        
        posteriorProbMatrix{ib} = posteriorProbMatrix{ib}./repmat(sum(posteriorProbMatrix{ib}, 1), [nPosBins, 1]); 
    end
    

    boutFirings_unit(currUnit, :) = cell2mat(boutFirings_unit_bout');
    boutFirings_otherUnits        = cell2mat(boutFirings_otherUnits');
    
    posteriorProbMatrix_allBouts  = cell2mat(posteriorProbMatrix');
    
    
    % only the time bins with at least one other active unit are valid for calculating learned tunings
    idx = sum(posteriorProbMatrix_allBouts, 1) > 0 & sum(boutFirings_otherUnits, 1) >= 1;
    

    if ~all(~idx)
        
        posteriorProbMatrix_allBouts = posteriorProbMatrix_allBouts(:, idx);
        px = mean(posteriorProbMatrix_allBouts, 2);

        weightedSummation = sum(repmat(boutFirings_unit(currUnit, idx), [nPosBins 1]) .* posteriorProbMatrix_allBouts, 2)/ntTimeBins;

        learnedTunings(currUnit, :) = weightedSummation ./ px; 
            
    else
        continue
    end   

end

end
