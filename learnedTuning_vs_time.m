function learnedTuning_vs_time(sessionNumber)


sz = getenv('SLURM_CPUS_PER_TASK');

theCluster = parcluster('local');

JobFolder = sprintf('/home/kmaboudi/.matlab/trash/job_%s', sessionNumber);
mkdir(JobFolder)
theCluster.JobStorageLocation = JobFolder;

p = parpool(theCluster, str2double(sz));



addpath(genpath('/nfs/turbo/umms-kdiba/NCMLproject/ReplayPreplayAnalyses'))

parentDir = '/nfs/turbo/umms-kdiba/Kourosh/NCMLproject/concat_GreatLakes_datasets_temp';
% parentDir = '/home/kouroshmaboudi/Documents/NCMLproject/concat_GreatLakes_datasets_temp/';


rr = dir(parentDir);
sessionName = rr(sessionNumber+2).name;

basePath    = fullfile(parentDir, sessionName);
storagePath = fullfile(basePath, 'learnedTunings');
mkdir(storagePath)


load(fullfile(basePath, 'BayesianDecoding', [sessionName '.PBEInfo_replayScores.mat']), 'PBEInfo_replayScores')
load(fullfile(basePath, 'spikes', [sessionName '.spikes.mat']), 'spikes_pyr', 'fileInfo')
load(fullfile(basePath, 'spikes', [sessionName '.clusterQuality.mat']), 'clusterQuality')


% behavioral epochs
behavior = fileInfo.behavior;

startT.pre  = behavior.time(1,1); endT.pre  = behavior.time(2,1); 
startT.run  = behavior.time(2,1); endT.run  = behavior.time(2,2); 
startT.post = behavior.time(2,2); endT.post = behavior.time(3,2); 


spikes = spikes_pyr;
nPosBins = numel(spikes(1).spatialTuning_smoothed.uni);


nUnits  = numel(spikes); 


% non-directional spatial tuning
spatialTunings_merge = nan(nUnits, nPosBins);
PF_prefPos = nan(nUnits, 1);
for iUnit = 1:nUnits
    spatialTunings_merge(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.uni;
    [~, PF_prefPos(iUnit)]         = max(spatialTunings_merge(iUnit, :), [], 2);
end


% we are going to analyze learned tuning separately for each epoch, so we
% need to reorganize the PBEs


% PBEInfo = PBEInfo(acceptedIdx); % only PBEs qulaified in terms of duration, number of participant units, the brain satte during which they occurred
PBEInfo = PBEInfo_replayScores;
PBEInfo = PBEInfo(ismember({PBEInfo.brainState}, {'QW'; 'NREM'}));

peakT   = [PBEInfo.peakT]';


%% learned tunings in 15 minutes sliding time windows
% Spatial information and correlation with place fields of the learned tunings


epochNames = {'pre'; 'run'; 'post'};

binDur  = 900; % 15 minutes
stepDur = 300; % 5 minutes


for iepoch = 1:3
    
    currEpoch = epochNames{iepoch};
    fprintf(['\nProcessing ' currEpoch ' ..'])
    
    
    allPBEs = PBEInfo(strcmp({PBEInfo.epoch}, currEpoch));
    tNum_PBEs = numel(allPBEs);
    
    posteriorProbMatrix = cell(tNum_PBEs, 1);
    for pbe = 1:tNum_PBEs
        posteriorProbMatrix{pbe} = allPBEs(pbe).posteriorProbMat;
    end
    posteriorProbMatrix = cell2mat(posteriorProbMatrix');

    virtualOccupancy = mean(posteriorProbMatrix, 2);
    virtualOccupancy = virtualOccupancy / sum(virtualOccupancy); % normalize


    
    totalT = endT.(currEpoch) - startT.(currEpoch);
    nBins  = floor((totalT - binDur)/stepDur) + 1;

    binStarts               = startT.(currEpoch) + (0:nBins-1)*stepDur;
    binEnds                 = binStarts + binDur;
    binCenters.(currEpoch)  = binStarts + binDur/2;
    
    
    selectPBEs = cell(nBins, 1);
    nPBEs.(currEpoch) = zeros(nBins, 1);
    for ibin = 1:nBins
        selectPBEs{ibin} = find(peakT >= binStarts(ibin) & peakT < binEnds(ibin));
        nPBEs.(currEpoch)(ibin) = numel(selectPBEs{ibin});
    end
    

    % actual PBEs
    fprintf('\nActual PBEs')
    
    learnedTuningCorrMat_time.(currEpoch)     = nan(nUnits, nUnits, nBins);
    learnedTuningPFcorr_time.(currEpoch)      = nan(nUnits, nBins);
    learnedTuningSpatialInfo_time.(currEpoch) = nan(nUnits, nBins);
    learnedTuningPFKLdiv_time.(currEpoch)     = nan(nUnits, nBins);
    learnedTuning_prefPos_time.(currEpoch)    = nan(nUnits, nBins);
    learnedTuningPF_prefPosDist.(currEpoch)   = nan(nUnits, nBins);
    
    currLearnedTunings = calLearnedTuning_vs_time(PBEInfo, spikes, selectPBEs, clusterQuality);
    
    
    learnedTunings_time.(currEpoch).data  = currLearnedTunings;
    for ibin = 1:nBins
        
        learnedTuningCorrMat_time.(currEpoch)(:, :, ibin) = corr(currLearnedTunings(:, :, ibin)', spatialTunings_merge'); 
        learnedTuningPFcorr_time.(currEpoch)(:, ibin)     = diag(learnedTuningCorrMat_time.(currEpoch)(:, :, ibin)); % the correlation of PF and learned tuning for a given unit

        meanFR = sum(repmat(virtualOccupancy', [nUnits 1]).* currLearnedTunings(:, :, ibin), 2) ./ sum(virtualOccupancy); 
        learnedTuningSpatialInfo_time.(currEpoch)(:, ibin) = sum(repmat(virtualOccupancy', [nUnits 1]).* (currLearnedTunings(:,:,ibin) ./ repmat(meanFR, [1 nPosBins])) .* log2(currLearnedTunings(:,:,ibin) ./ repmat(meanFR, [1 nPosBins])), 2);
    
        KLdiv = calKLDivergence(currLearnedTunings(:, :, ibin)', spatialTunings_merge');
        learnedTuningPFKLdiv_time.(currEpoch)(:, ibin) = diag(KLdiv);

        [learnedTuningPF_prefPosDist.(currEpoch)(:, ibin), learnedTuning_prefPos_time.(currEpoch)(:, ibin)] = calDistPrefPosition(currLearnedTunings(:, :, ibin), PF_prefPos);
    
    end
   
end


fileName = [sessionName '.learnedTuning_vs_time.mat'];

save(fullfile(storagePath, fileName), 'nPBEs', 'binCenters', ...
        'learnedTunings_time', 'learnedTuningCorrMat_time', 'learnedTuningPFcorr_time', ...
        'learnedTuningSpatialInfo_time', ...
        'learnedTuningPFKLdiv_time', ...
        'learnedTuning_prefPos_time', 'learnedTuningPF_prefPosDist', '-v7.3')



end

%% sub-functions


function KLdiv = calKLDivergence(learnedTunings, spatialTunings)


    % the individual learned tunings and spatial tunings should be in columns
    % this function works well when the learnedTunings matrix contains multipels instance of learned tunings (for examples if they belong to different time windows or different instances of unit identity shuffle)
    
    
    nPosBins = size(spatialTunings, 1);
    
    spatialTunings = spatialTunings ./ repmat(sum(spatialTunings, 1), [nPosBins 1]);
    spatialTunings = spatialTunings + eps;
    
    learnedTunings = learnedTunings ./ repmat(sum(learnedTunings, 1), [nPosBins 1]);
    learnedTunings = learnedTunings + eps;
    
    
    nSTs = size(spatialTunings, 2);
    nLTs = size(learnedTunings, 2); % number of learned tunings
    
    
    KLdiv = nan(nLTs, nSTs);
    for ii = 1:nSTs
        
        currSpatialTuning = spatialTunings(:, ii);
       
        spatialTuningTerm = repmat(currSpatialTuning, [1 nLTs]);
        KLdiv(:, ii) = sum(spatialTuningTerm .* (log(spatialTuningTerm) - log(learnedTunings)), 1);
    
    end


end


function [prefPosDist, asTuning_prefPos] = calDistPrefPosition(learnedTunings, PF_prefPos)  
    
    nUnits     = numel(PF_prefPos);
    nInstances = size(learnedTunings, 3);
    nPosBins   = size(learnedTunings, 2);
    
    prefPosDist      = nan(nUnits, nInstances);
    asTuning_prefPos = nan(nUnits, nInstances);
    
    for inst = 1:nInstances
       currAsTunings = learnedTunings(:,:, inst);
       
       [~, asTuning_prefPos(:, inst)] = max(currAsTunings, [], 2);
       prefPosDist(:, inst)           = abs(asTuning_prefPos(:, inst) - PF_prefPos(:));
       prefPosDist(:, inst)           = prefPosDist(:, inst)/nPosBins;
       
    end

end

