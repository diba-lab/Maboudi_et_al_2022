function learnedTuning_brainStates(sessionNumber, condition)


parentDir = '/home/kouroshmaboudi/Documents/NCMLproject';

rr = dir(fullfile(parentDir, 'learnedTuning_finalResults'));
sessionName = rr(sessionNumber+2).name


basePath    = fullfile(parentDir, 'learnedTuning_finalResults', sessionName);
storagePath = fullfile(basePath, 'learnedTunings');
mkdir(storagePath)

load(fullfile(basePath, 'spikes', [sessionName '.spikes.mat']), 'spikes_pyr', 'fileInfo')
load(fullfile(basePath, 'spikes', [sessionName '.clusterQuality.mat']), 'clusterQuality')



% load the brain state information
fileName = fullfile(parentDir, 'StateDetectionResults',  sessionName, [sessionName '.brainStateDetection_HMMtheta_EMG_SWS_SchmidtTrigger.mat']);

if isfile(fileName)
    load(fileName, 'brainState')
    brainState = brainState.bouts(:, 1:3);
    
    % to comply with Hiro's datasets
    % 1:NREM, 2:REM, 3:QWAKE, 4:WAKE
    
    temp = brainState;
    temp(brainState(:, 3) == 3, 3) = 4;
    temp(brainState(:, 3) == 4, 3) = 3;
    brainState = temp;

    durations = diff(brainState(:, 1:2), [], 2);
    inclusionIdx = durations > 6;

    brainState = brainState(inclusionIdx, :);

else % for Hiro's datasets
    brainState = fileInfo.brainStates;

end



% population burst events

load(fullfile(basePath, 'PopulationBurstEvents', [sessionName '.PBEs_before_splitting.mat']), 'primaryPBEs')

PBEs(:,1) = [primaryPBEs.startT];
PBEs(:,2) = [primaryPBEs.endT];


% behavioral epochs

behavior = fileInfo.behavior;

startT.pre  = behavior.time(1,1); endT.pre  = behavior.time(2,1); 
startT.run  = behavior.time(2,1); endT.run  = behavior.time(2,2); 
startT.post = behavior.time(2,2); endT.post = startT.post + 5*60*60; %behavior.time(3,2); 

 

spikes   = spikes_pyr;
nPosBins = numel(spikes(1).spatialTuning_smoothed.uni);
nUnits   = numel(spikes); 


% non-directional spatial tuning

spatialTunings = nan(nUnits, nPosBins);
PFpeakLoc      = nan(nUnits, 1);

for iUnit = 1:nUnits
    spatialTunings(iUnit, :) = spikes(iUnit).spatialTuning_smoothed.uni; 
    [~, PFpeakLoc(iUnit)] = max(spatialTunings(iUnit, :));

end


timeStamps = 0:0.05:endT.post;


brainState(:,2) = brainState(:, 2) - 1e-3;
brainState(:,4) = brainState(:, 3);

timePnts = brainState(:, 1:2)';
timePnts = timePnts(:);

stateIdx = brainState(:, 3:4)';
stateIdx = stateIdx(:);

stateIdx_intrp = interp1(timePnts, stateIdx, timeStamps);


% population burst events
PBEidx = zeros(numel(timeStamps), 1);

for ipbe = 1:size(PBEs, 1)
    PBEidx(timeStamps > PBEs(ipbe, 1) & timeStamps < PBEs(ipbe, 2)) = 1;
end



% bouts with the desired brain state/condition

% condition: 1:PBEs, 2:REM, 3:NREM non-PBE, 4:home cage awake theta

switch condition 

    case 1
        conditionIdx = PBEidx == 1;
        conditionName = 'PBEs';

    case 2 % REM
        conditionIdx = stateIdx_intrp' == 2 & PBEidx == 0;
        conditionName = 'REM';

    case 3 % NREM non-PBE low ripple amplitude
        conditionIdx = stateIdx_intrp' == 1 & PBEidx == 0;
        conditionName = 'nonPBEsExtend_NREM';

    case 4 % active wake
        conditionIdx = stateIdx_intrp' == 4 & PBEidx == 0;
        conditionName = 'aWAKE';
    
    case 5 % whole NREM
        conditionIdx = stateIdx_intrp' == 1;
        conditionName = 'NREM';

end
 


% the boundaries of each state

try 
    crossup   = find([0; diff(conditionIdx)] == 1);
    crossdown = find([0; diff(conditionIdx)] == -1);
catch 
    crossup   = find([0 diff(conditionIdx)] == 1);
    crossdown = find([0 diff(conditionIdx)] == -1);
end


if crossdown(1) < crossup(1); crossdown(1) = []; end
if crossup(end) > crossdown(end); crossup(end) = []; end

bouts = timeStamps([crossup crossdown]);


%% learned Tunings 
% place field fidelity and spatial information of the learned tunings 


epochNames = {'pre'; 'run'; 'post'};

for binDur = 0.02
    
    for iEpoch = 1:3

        currEpoch = epochNames{iEpoch};
        epoch     = [startT.(currEpoch) endT.(currEpoch)]; 

        currEpoch_includeIdx = bouts(:, 1) > epoch(1) & bouts(:, 1) < epoch(2); 
        currBouts = bouts(currEpoch_includeIdx, :);

        fprintf(['\nProcessing ' currEpoch ' ,binDur = ' num2str(binDur) ' ..'])

        
        % actual data
        fprintf('\nCalculating the learned tunings based on actual PBEs ..')
        
        learnedTunings.(currEpoch) = calculateAssemblyTuning_selectPeriods(currBouts, spikes, binDur, clusterQuality);
        
        currLearnedTunings = learnedTunings.(currEpoch);
        
        learnedTuningPFcorr.(currEpoch)         = nan(nUnits, 1);
        learnedTuningPFKLdiv.(currEpoch)        = nan(nUnits, 1);
        learnedTuningPF_prefPosDist.(currEpoch) = nan(nUnits, 1);
        learnedTuning_prefPos.(currEpoch)       = nan(nUnits, 1);

        allCorrs = corr(currLearnedTunings(:, :, is)', spatialTunings');
        learnedTuningPFcorr.(currEpoch)(:, is) = diag(allCorrs);

        allKLdiv = calKLDivergence(currLearnedTunings(:, :, is)', spatialTunings');
        learnedTuningPFKLdiv.(currEpoch)(:, is) = diag(allKLdiv);
        
        [learnedTuningPF_prefPosDist.(currEpoch)(:, is), learnedTuning_prefPos.(currEpoch)(:, is)] = ...
            calDistPrefPosition(currLearnedTunings(:, :, is), PFpeakLoc, []);

    end


    fileName = sprintf('%s.assemblyTunings_%s.mat', sessionName, conditionName);

    save(fullfile(storagePath, fileName), ...
        'learnedTunings', ...
        'learnedTuning_prefPos', ...
        'learnedTuningPF_prefPosDist', ...
        'learnedTuningPFcorr', 'learnedTuningPFKLdiv'); 

end



end



%% subfunctions

function KLdiv = calKLDivergence(learnedTunings, spatialTunings)

% The individual units should be in columns of learned tunings or spatialTunings
% matrix

spatialTunings = spatialTunings ./ repmat(sum(spatialTunings, 1), [size(spatialTunings, 1) 1]);
spatialTunings = spatialTunings + eps;

[~, nUnits] = size(spatialTunings);


learnedTunings = learnedTunings ./ repmat(sum(learnedTunings, 1), [size(learnedTunings, 1) 1]);
learnedTunings = learnedTunings + eps;


KLdiv = nan(nUnits);

for iUnit = 1:nUnits
    currSpatialTuning = spatialTunings(:, iUnit);
    
    spatialTuningTerm = repmat(currSpatialTuning, [1 nUnits]);
    KLdiv(:, iUnit) = sum(spatialTuningTerm .* (log(spatialTuningTerm) - log(learnedTunings)), 1);
end

end


function [prefPosDist, asTuning_prefPos] = calDistPrefPosition(assemblyTunings, PF_prefPos, activeUnits)  
    
    nUnits = numel(PF_prefPos);
    nSets = size(assemblyTunings, 3);
    nPosBins = size(assemblyTunings, 2);
    
    if isempty(activeUnits)
        activeUnits = 1:nUnits;
    end
    
    prefPosDist   = nan(nUnits, nSets);
    asTuning_prefPos = nan(nUnits, nSets);

    for n = 1:nSets
       currAsTunings = assemblyTunings(:,:, n);
       [~, asTuning_prefPos(:, n)] = max(currAsTunings, [], 2);
       prefPosDist(activeUnits, n) = abs(asTuning_prefPos(activeUnits, n) - PF_prefPos(activeUnits));
       prefPosDist(activeUnits, n) = prefPosDist(activeUnits, n)/nPosBins;
    end

end
