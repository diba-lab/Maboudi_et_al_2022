function learnedTuning_mainCode(sessionNumber)


parentDir = '/nfs/turbo/umms-kdiba/Kourosh/NCMLproject/concat_GreatLakes_datasets_temp';
% parentDir = '/data/concat_GreatLakes_datasets_temp';


rr = dir(parentDir);
sessionName = rr(sessionNumber+2).name


sz = getenv('SLURM_CPUS_PER_TASK');

theCluster = parcluster('local');

JobFolder = sprintf('/home/kmaboudi/.matlab/trash2/job_%s', sessionNumber);
mkdir(JobFolder)
theCluster.JobStorageLocation = JobFolder;


p = parpool(theCluster, str2double(sz));


basePath    = fullfile(parentDir, sessionName);
storagePath = fullfile(basePath, 'assemblyTunings');
mkdir(storagePath)


load(fullfile(basePath, 'BayesianDecoding', [sessionName '.PBEInfo_replayScores.mat']), 'PBEInfo_replayScores') % The PBEs (ripples) with different kind of replay scores calculated for each PBE
load(fullfile(basePath, 'spikes', [sessionName '.spikes.mat']), 'spikes_pyr', 'fileInfo') % the spike info of all pyramidal units
load(fullfile(basePath, 'spikes', [sessionName '.clusterQuality.mat']), 'clusterQuality') % the cluster quality info, containing the pairwise L-ratios 


spikes = spikes_pyr;
nPosBins = numel(spikes(1).spatialTuning_smoothed.uni);


nUnits  = numel(spikes); 



% non-directional spatial tuning
spatialTunings_merge = zeros(nUnits, nPosBins);
for iUnit = 1:nUnits
    spatialTunings_merge(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.uni;
end

epochs = fileInfo.behavior.time;

% we need to analyze the learned tuning separately for each epoch, so we
% need to reorganize the PBEs


PBEInfo = PBEInfo_replayScores;

epochNames = {'pre', 'run', 'post'};

for iepoch = 1:numel(epochNames) 
    currEpoch = epochNames{iepoch};
    
    PBEs.(currEpoch)  = PBEInfo(strcmp({PBEInfo.epoch}, currEpoch));
    if iepoch == 3
        peaks = [PBEs.(currEpoch).peakT];
        idx  = peaks < (epochs(3,1)+4*60*60); % limiting the post sleep epoch to the first 4 hours
        PBEs.(currEpoch) = PBEs.(currEpoch)(idx);
    end
    nPBEs.(currEpoch) = numel(PBEs.(currEpoch));
end


%% learned Tunings considering all of the PBEs
% Spatial information and correlation with place fields of assesmbly tunings


for iepoch = 1:3
    
    currEpoch = epochNames{iepoch};
    
    fprintf(['/nProcessing ' currEpoch ' ..'])
    
    
    currPBEs = PBEs.(currEpoch);
    
    
    posteriorProbMatrix = cell(nPBEs.(currEpoch), 1);
    for pbe = 1:nPBEs.(currEpoch) 
        posteriorProbMatrix{pbe} = currPBEs(pbe).posteriorProbMat;
    end
    posteriorProbMatrix = cell2mat(posteriorProbMatrix');
   
    virtualOccupancy.(currEpoch) = mean(posteriorProbMatrix, 2);
    virtualOccupancy.(currEpoch) = virtualOccupancy.(currEpoch) / sum(virtualOccupancy.(currEpoch)); % normalize
    
    
    
    % Actual PBEs
    
    fprintf('\nCalculating the learned tunings based on actual PBEs ..')
    
    selectPBEs = [];
    currLearnedTunings = calLearnedTunings(currPBEs, spikes, selectPBEs, clusterQuality);
    
    
    learnedTunings.(currEpoch).data = currLearnedTunings;
    
    learnedTuningCorrMat.(currEpoch).data = corr(currLearnedTunings', spatialTunings_merge'); 
    learnedTuningPFcorr.(currEpoch).data  = diag(learnedTuningCorrMat.(currEpoch).data); % the correlation of PF and assembly tuning for a given unit
    
    
    meanFR = sum(repmat(virtualOccupancy.(currEpoch)', [nUnits 1]).* currLearnedTunings, 2) ./ sum(virtualOccupancy.(currEpoch)); % The calculation of mean firing rate could be a bit questionable
    learnedTuningSpatialInfo.(currEpoch).data = sum(repmat(virtualOccupancy.(currEpoch)', [nUnits 1]).* (currLearnedTunings ./ repmat(meanFR, [1 nPosBins])) .* log2(currLearnedTunings ./ repmat(meanFR, [1 nPosBins])), 2);

    
end


fileName = [sessionName '.assemblyTunings_allPBEs.mat']; 

save(fullfile(storagePath, fileName), ...
    'nPBEs', ...
    'learnedTunings', ...
    'learnedTuningCorrMat', ...
    'learnedTuningPFcorr', ...
    'learnedTuningSpatialInfo')



%% How learnedTuning depend on BD replay score

[learnedTunings_sub, learnedTuningCorrMat_sub, learnedTuningPFcorr_sub, ...
    learnedTuningSpatialInfo_sub] = deal(struct('pre', [], 'run', [], 'post', [])); % assembly tunings calculated based on PBEs with replay scores in different quartiles of the replay score distributions

nPBEs_subset = struct('pre', [], 'run', [], 'post', []);

                  
replayScoreMethods = {'rt_ui'; 'wc_ui'; 'wc_ts'; 'wc_ds'; 'rt_ds'};
                      

replayScoreMethods_fullName = {'radon Integral - unit ID shuffle'; ...
                               'weighted Corr - unit ID shuffle' ; ...
                               'weighted Corr - wPBE time swap';
                               'weighted Corr - column cycle shuffle';
                               'radon Integral - column cycle shuffle'};
                           
fileName = [sessionName '.learnedTuning_vs_replayScores.mat'];
                          
                           
for iepoch = 1:3
    
    currEpoch = epochNames{iepoch};
    currPBEs = PBEs.(currEpoch);

    posteriorProbMatrix = cell(nPBEs.(currEpoch), 1);
    for pbe = 1:nPBEs.(currEpoch) 
        posteriorProbMatrix{pbe} = currPBEs(pbe).posteriorProbMat;
    end
    posteriorProbMatrix = cell2mat(posteriorProbMatrix');


    virtualOccupancy.(currEpoch) = mean(posteriorProbMatrix, 2);
    virtualOccupancy.(currEpoch) = virtualOccupancy.(currEpoch) / sum(virtualOccupancy.(currEpoch)); % normalize 
    
end
                               

for irsm = 1:numel(replayScoreMethods)
    
    replayScoreMethod = replayScoreMethods{irsm};
    replayScoreMethod_name = replayScoreMethods_fullName{irsm};
    
    fprintf(['/nReplay score method: ' replayScoreMethod_name ' ..'])
    
    
    for iepoch = 1:3 
        
        currEpoch = epochNames{iepoch};
        
        fprintf(['/nepoch ' currEpoch ' ..'])
        
        
        
        currBDscores = [PBEs.(currEpoch).(replayScoreMethod)];

        nPBEs_subset.(currEpoch).(replayScoreMethod) = zeros(4,1);

        learnedTunings_sub.(currEpoch).data.(replayScoreMethod)      = zeros(nUnits, nPosBins, 4);        
        learnedTunings_sub.(currEpoch).ui.(replayScoreMethod)        = cell(4,1);

        learnedTuningSpatialInfo_sub.(currEpoch).data.(replayScoreMethod)  = nan(nUnits, 4);
        learnedTuningPFcorr_sub.(currEpoch).data.(replayScoreMethod)       = nan(nUnits, 4);
        learnedTuningCorrMat_sub.(currEpoch).data.(replayScoreMethod)      = nan(nUnits, nUnits, 4);     


        for iPBEset = 1:4


            fprintf('\nCalculating learned tunings based on PBEs in quartile %d of replay score ..', iPBEset)


            selectPBEs = find(currBDscores' > (iPBEset-1)*25 & currBDscores' <= iPBEset*25); 

            nPBEs_subset.(currEpoch).(replayScoreMethod)(iPBEset) = numel(selectPBEs);
            currbinnedPBEs = PBEs.(currEpoch);
            
            
            % calculate the assembly tunings based on the current PBEs
            
            currLearnedTunings = calLearnedTunings(currbinnedPBEs, spikes, selectPBEs, clusterQuality);

            learnedTunings_sub.(currEpoch).data.(replayScoreMethod)(:, :, iPBEset) = currLearnedTunings ;


            meanFR  = sum(repmat(virtualOccupancy.(currEpoch)', [nUnits 1]).* currLearnedTunings, 2) ./ sum(virtualOccupancy.(currEpoch));
            learnedTuningSpatialInfo_sub.(currEpoch).data.(replayScoreMethod)(:, iPBEset) = sum(repmat(virtualOccupancy.(currEpoch)', [nUnits 1]).* (currLearnedTunings ./ repmat(meanFR, [1 nPosBins])) .* log2(currLearnedTunings ./ repmat(meanFR, [1 nPosBins])), 2);

            learnedTuningCorrMat_sub.(currEpoch).data.(replayScoreMethod)(:, :, iPBEset) = corr(permute(currLearnedTunings, [2 1]), spatialTunings_merge');
            learnedTuningPFcorr_sub.(currEpoch).data.(replayScoreMethod)(:, iPBEset)     = diag(learnedTuningCorrMat_sub.(currEpoch).data.(replayScoreMethod)(:, :, iPBEset));

        end

    end
    
    
end



fileName = [sessionName '.assemblyTuning_vs_replayScores.mat'];

save(fullfile(storagePath, fileName), ...
    'nPBEs_subset', ...
    'learnedTunings_sub', ...
    'learnedTuningCorrMat_sub', ...
    'learnedTuningPFcorr_sub', ...
    'learnedTuningSpatialInfo_sub')


end
