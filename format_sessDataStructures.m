function format_sessDataStructures(sessionNumber)


parentDir = '/home/kouroshmaboudi/Documents/NCMLproject';

addpath(genpath(fullfile(parentDir, '/ReplayPreplayAnalyses')))

datasetDir = fullfile(parentDir, '/Bapun_NSD_datasets');


stateDetectionResultsDir = fullfile(parentDir, 'StateDetectionResults');

outputDir = fullfile(parentDir, 'assemblyTuning_finalResults');
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

sessionNames = {'RatN_Day2_2019-10-11_03-58-54'; ...
    'RatS-Day2-2020-11-27_10-22-29'; ...
    'RatU_Day2NSD_2021-07-24_08-16-38'; ...
    'RatV_Day1NSD_2021-10-02_08-10-23'; ...
    'RatV_Day3NSD_2021-10-07_08-10-12'};



% MAZE SHAPES
mazeLimits = [-inf inf -inf 128; ... % RatN
              -inf inf -inf inf; ... % Rat S
              -inf inf -inf inf; ... % Rat U
              -inf inf -inf inf; ... % Rat V - Day1
              -inf inf -inf inf];    % Rat V - Day3

% mazeShapes = {'L-shape'; 'L-shape'; 'circular'; 'linear'; 'circular'};


% CURRENT SESSION        
sessionName = sessionNames{sessionNumber};



%% loading session data

baseFile = fullfile(datasetDir, sessionName, sessionName);


% load spike data
load([baseFile '.spike.mat'], 'spikes');



% load behavioral paradigm data
load([baseFile '.epochs.mat'], 'pre', 'maze', 'post', 're_maze')




% load position data

% 2D positions 
s = load([baseFile '.position.mat'], 'traces', 'sampling_rate', 't_start'); 
positions             = s.traces';
position_samplingRate = double(s.sampling_rate);
ts_Pos                = double(s.t_start);

% lineared maze positions
s = load([baseFile '.maze.linear.mat'], 'traces', 't_start'); % sampling rate is the same as for 2D
linearized_MAZE       = s.traces';
ts_MAZE_Pos           = double(s.t_start);  

% linearized remaze positions
s = load([baseFile '.remaze.linear.mat'], 'traces', 't_start'); 
linearized_REMAZE     = s.traces';
ts_REMAZE_Pos         = double(s.t_start);  


% a unified linear position variable

tpnts     = ts_Pos + (1:length(positions)) / position_samplingRate;
linearPos = nan(numel(tpnts), 1);


startIdx  = find(tpnts > ts_MAZE_Pos, 1, 'first'); 
linearPos(startIdx: startIdx+length(linearized_MAZE)-1) = linearized_MAZE;

startIdx  = find(tpnts > ts_REMAZE_Pos, 1, 'first'); 
linearPos(startIdx: startIdx+length(linearized_REMAZE)-1) = linearized_REMAZE;



storageDir = fullfile(outputDir,  sessionName);
if ~exist(storageDir, 'dir'); mkdir(storageDir); end

try
    behavior.time = double([pre; maze; post; re_maze]);
catch
    behavior.time = double([pre; maze1; post; re_maze]);
end



% give a warning if there is a gap between the end of one epoch and start
% of the next 

if behavior.time(2,1) ~= behavior.time(1,2) || behavior.time(3,1) ~= behavior.time(2,2) 
    warning('There are gaps between the epochs')
end



%% BRAIN STATES / THETA-NONTHETA

% We are not going to calculate the periods belonging to each brain state, as the calculated boundaries lacked enough accuracy for the purpose of PBE detection. Instead we use
% [almost instantaneous] estimations of theta/delta ratio and slow wave amplitude to estimate the state in which each population burst event occurred  


% loading theta info
filename = sprintf([sessionName '.brainStateDetection_HMMtheta_EMG_SWS_SchmidtTrigger.mat']);
load(fullfile(stateDetectionResultsDir, sessionName, filename), 'timePnts', 'theratio'); 
brainState.thetaTimePnts = timePnts;
brainState.theratio      = theratio;


% loading slow wave power/slope 
filename = sprintf([sessionName '.SleepScoreMetrics.mat']);
load(fullfile(stateDetectionResultsDir, sessionName, filename), 'SleepScoreMetrics')


brainState.slowWave        = SleepScoreMetrics.broadbandSlowWave;
brainState.sw_emg_timePnts = SleepScoreMetrics.t_clus;

try 
  load(fullfile(stateDetectionResultsDir, sessionName, sprintf([sessionName '.swthresh.mat'])), 'swthresh'); 
    brainState.swthresh = swthresh;
catch
    brainState.swthresh = SleepScoreMetrics.histsandthreshs.swthresh;
end



% loading emg 

brainState.emg = SleepScoreMetrics.EMG;

try
    load(fullfile(stateDetectionResultsDir, sessionName, sprintf([sessionName '.emgThresh.mat'])), 'emgThresh');
    brainState.emgThresh = emgThresh;
catch
    brainState.emgThresh = SleepScoreMetrics.histsandthreshs.EMGthresh; 
end



% bouts with different brain states

filename = sprintf([sessionName '.brainStateDetection_HMMtheta_EMG_SWS_SchmidtTrigger.mat']);
s = load(fullfile(stateDetectionResultsDir, sessionName, filename), 'brainState'); 
brainState.periods = s.brainState.bouts;
brainState.names   = s.brainState.names;



% bad channels to exclude for lpf event detections (e.g., ripples)

filename = sprintf([sessionName '.sessionInfo.mat']);
load(fullfile(stateDetectionResultsDir, sessionName, filename), 'sessionInfo')

if isfield(sessionInfo, 'badchannels')
    badChannels = sessionInfo.badchannels;
else
    badChannels = [];
end



%% SESSION'S INFORMATION


% load spikes and fileInfo
folderName = fullfile(storageDir,  'spikes');
fileName   = fullfile(folderName, [sessionName '.spikes.mat']);
overwrite  = 0;


if exist(fileName, 'file') && overwrite == 0

    load(fileName, 'spikes', 'spikes_pyr', 'fileInfo')

else
    load(fileName,  'fileInfo')
    mkdir(folderName); 

    Par = LoadXml(baseFile); % Load the recording configurations from the xml file

    fileInfo.sessionName   = sessionName;
    fileInfo.animal        = sessionName(1:strfind(sessionName, '_')-1);
    fileInfo.xyt           = [];
    fileInfo.linearPos     = []; % linearized position and lap information
    fileInfo.speed         = [];
    fileInfo.tbegin        = behavior.time(1,1);
    fileInfo.tend          = behavior.time(4,2);
    fileInfo.Fs            = Par.SampleRate; 
    fileInfo.lfpSampleRate = Par.lfpSampleRate;
    fileInfo.timeUnit      = 1; % in sec
    fileInfo.nCh           = Par.nChannels;
    fileInfo.badChannels   = badChannels;
    fileInfo.behavior      = behavior;
    fileInfo.brainState    = brainState;
   
        
    % ripples
    if ~isfield(fileInfo, 'RippleChannels')
        best_channels = BigRippleChannels(baseFile, fileInfo);
        fileInfo.RippleChannels = best_channels; % check if there are no bad channels among the best_channels 
        save(fileName, 'fileInfo') % the value of fileInfo will be updated later
    end
    
    positions = positions';
    y = positions(1,:)';
    x = positions(2,:)';
    position_samplingRate = 60; % RatV
    time = (0:1/position_samplingRate:(length(positions)-1)/position_samplingRate)';


    fileInfo.xyt = [x y time];       

    speed = calspeed(fileInfo, position_samplingRate); % animal's velocity
    speed.v(speed.t < behavior.time(1,2)) = 0;
    
    fileInfo.speed = speed;
    


    %% LINEARIZE POSTION AND CALCULATE LAPS

    fileInfo.xyt(x > mazeLimits(sessionNumber, 2) | x < mazeLimits(sessionNumber, 1) | y > mazeLimits(sessionNumber, 4) | y < mazeLimits(sessionNumber, 3), :) = [];
    
    fileInfo.linearPos(:, 1) = linearPos; 
    fileInfo.linearPos(:, 2) = fileInfo.xyt(:, 3); % time


    direction = 'bi';
    [lapsStruct, turningPeriods] = calculateLapTimings(fileInfo, direction, storageDir); 

    if length(lapsStruct.RL) > length(lapsStruct.LR)
       lapsStruct.RL(1,:) = [];
    end

    totNumLaps = size(lapsStruct.RL, 1) + size(lapsStruct.LR, 1);
    laps = zeros(totNumLaps, 2);
    laps(1:2:totNumLaps, :)  = lapsStruct.LR;
    laps(2:2:totNumLaps, :)  = lapsStruct.RL;


    laps(:, 3) = 1:size(laps, 1); 



    fileInfo.linearPos(:, 3) = zeros(size(linearPos)); % labeling the postion samples with the calculated laps (if not part of any lap the label is zero)

    for ii = 1: length(laps)
       idx =  fileInfo.linearPos(:, 2) > laps(ii, 1) & fileInfo.linearPos(:, 2) < laps(ii, 2);
       fileInfo.linearPos(idx, 3) = laps(ii, 3);
    end

    runSpeedThresh = 10; % cm/s
    

    
    %% calculating position, velocity, etc at each spike (these will be needed for calculation of place fields)

%     spikes = spike; % just to make the variable names the same across different datasets
    nUnits = numel(spikes);


    stabilityPeriod_pre  = [max(behavior.time(1,2)-3*60*60, behavior.time(1,1)) behavior.time(1,2)];
    stabilityPeriod_maze =  behavior.time(2,:);
    stabilityPeriod_post = [behavior.time(3,1) behavior.time(3,1)+3*60*60];

    stb = [stabilityPeriod_pre; stabilityPeriod_maze; stabilityPeriod_post];


    for unit = 1:nUnits

%         spikes(unit).time = spikes(unit).t';
        spikes(unit).time = spikes(unit).time;
        spikeTimes = spikes(unit).time;
        spikes(unit).x = interp1(fileInfo.xyt(:, 3), fileInfo.xyt(:, 1), spikeTimes);
        spikes(unit).y = interp1(fileInfo.xyt(:, 3), fileInfo.xyt(:, 2), spikeTimes);
        spikes(unit).linearPos = interp1(fileInfo.linearPos(:, 2), fileInfo.linearPos(:, 1), spikeTimes);

        spikes(unit).speed = interp1(speed.t, speed.v, spikeTimes);

        spikes(unit).lap = zeros(1, numel(spikeTimes));
        for spk = 1:numel(spikeTimes)
            index = find(laps(:, 1) < spikeTimes(spk) & laps(:, 2) > spikeTimes(spk));
            if ~isempty(index)
                spikes(unit).lap(spk) = laps(index, 3);
            end
        end

%         spikes(unit).id = [spikes(unit).shank spikes(unit).id];
%         spikes(unit).quality = spikes(unit).q;


        %
        spikes(unit).fr_by_epoch = nan(1,3); 
        for ip = 1:size(stb, 1)
           spikes(unit).fr_by_epoch(ip) = numel(find(spikes(unit).time > stb(ip, 1) & spikes(unit).time <= stb(ip, 2)))/diff(stb(ip, :));
        end

        spikes(unit).fr_pre_post_ratio = spikes(unit).fr_by_epoch(3)/ spikes(unit).fr_by_epoch(1); % the difference between the pre and post epoch next the the maze is enough


        if spikes(unit).fr_pre_post_ratio > 1
            spikes(unit).fr_pre_post_bias = 1; % smaller as a percent of the greater firing rate
            spikes(unit).fr_pre_post_ratio = 1/spikes(unit).fr_pre_post_ratio;
        else
            spikes(unit).fr_pre_post_bias = -1;
        end


        if spikes(unit).fr_pre_post_ratio > 0.3 && spikes(unit).fr_by_epoch(2) > 0.02 && strcmp(spikes(unit).group, 'good')
            spikes(unit).fr_isStable = 1;
        else
            spikes(unit).fr_isStable = 0;
        end

    end


    %% Spatial tuning of the units
    
    spikes_pyr = spikes(strcmp({spikes.group}, 'pyr  '));
    
    fileInfo.okUnits = 1:numel(spikes_pyr); % all pyramidal units regardless of their stability will be considered in Bayesian decoding


    close all

    subfolder = fullfile(storageDir, 'placeFields');
    if ~exist(subfolder, 'dir') 
        mkdir(subfolder)
    end

    %%% 1D spatial tuning: using linearized position

    posBinSize = 2;

    spikes_pyr = spatialTuning_1D_June21(spikes_pyr, behavior, [], [], fileInfo.speed, runSpeedThresh, 'LR', posBinSize, fileInfo, subfolder);
    spikes_pyr = spatialTuning_1D_June21(spikes_pyr, behavior, [], [], fileInfo.speed, runSpeedThresh, 'RL', posBinSize, fileInfo, subfolder);

    spikes_pyr = spatialTuning_1D_June21(spikes_pyr, behavior, [], [], fileInfo.speed, runSpeedThresh, 'uni', posBinSize, fileInfo, subfolder);


    close all
    
    
    % save spikes and file info
    fileName = fullfile(folderName, [sessionName '.spikes.mat']);
    save(fileName, 'spikes', 'spikes_pyr', 'fileInfo', '-v7.3') 

    
end



%% generate a clu and res file for interneurons and mua for visualizing population bursts on Neuroscope

% subfolder = fullfile(storageDir, 'placeFields');
% if ~exist(subfolder, 'dir') 
%     mkdir(subfolder)
% end
% 
% 
% if sessionNumber == 3
%     int_and_MUA_Idx = find(ismember({spikes.group}, {'inter'; 'mua'}));
% else
%     int_and_MUA_Idx = find(ismember([spikes.quality], [6 8]));
% end
% 
% res_t = [];
% clu_t = [];
% 
% for unit = 1:numel(int_and_MUA_Idx)
%     
%     spikeTimes = floor(spikes(int_and_MUA_Idx(unit)).time*fileInfo.Fs);
%     cluIdx     = unit*ones(numel(spikeTimes), 1);
%     
%     res_t = [res_t; spikeTimes];
%     clu_t = [clu_t; cluIdx];
%     
% end
% 
% [res_t, ind] = sort(res_t);
% clu_t = clu_t(ind)-1;  
% 
% Saveres(fullfile(subfolder, [fileInfo.sessionName 'mua.201'  '.res.3']), res_t);
% SaveClu(fullfile(subfolder, [fileInfo.sessionName 'mua.201'  '.clu.3']), clu_t);
% 


%% RIPPLE DETECTION

folderName = fullfile(storageDir,  'ripples');
fileName   = fullfile(folderName, [fileInfo.sessionName '.rippleEvents.mat']);

if exist(fileName, 'file')
    load(fileName, 'rippleLFP', 'rplDetectParams', 'rippleEvents')
end

overwrite = 0;
if ~exist('rippleLFP', 'var') || overwrite == 1
    
    mkdir(folderName);
    
    rplDetectParams.method                 = 'summedChan';
    rplDetectParams.threshold              = 2;

    rplDetectParams.minDur                 = 0.04; % number of bins
    rplDetectParams.maxDur                 = 0.60;
    rplDetectParams.binDur                 = 0.02;
    
    rplDetectParams.thRatioThresh          = 1;
    rplDetectParams.rippleAThresh_low      = 1;
    rplDetectParams.rippleAThresh_high     = 3;
    
    rplDetectParams.exclude               = brainState.periods(ismember(brainState.periods(:, 3), [2 3 4]), 1:2); % NREM and QW periods
    
    dur = diff(rplDetectParams.exclude');
    rplDetectParams.exclude(dur<0.5, :) = [];
    
    
    [ripplePeriods, rawLFP, ripplePower_summed, includeIdx] = RippleDetect_include(baseFile, fileInfo, rplDetectParams); % the results in seconds
    

    % downsample the ripple band lfp or power to 1kHz
    
    ts    = 1/fileInfo.lfpSampleRate;

    tend  = length(rawLFP)/fileInfo.lfpSampleRate;
    tPnts = ts:ts:tend;

    t_1ms    = 1e-3:1e-3:tend;

    
    rawLFP2   = zeros(numel(t_1ms), size(rawLFP, 2));
    for ich = 1:size(rawLFP, 2)
        rawLFP2(:, ich)   = interp1(tPnts, rawLFP(:, ich), t_1ms);
    end
    rawLFP = rawLFP2;
    
    
    power_summed = interp1(tPnts, ripplePower_summed, t_1ms);
   
    rippleLFP.rawLFP       = rawLFP;
    rippleLFP.power_summed = power_summed;
    rippleLFP.timePnts     = t_1ms;
      
end


%% DETECTING POPULATION BURST EVENTS (PBEs)

folderName = fullfile(storageDir,  'PopulationBurstEvents');
overwrite = 1;

if exist(folderName, 'dir') && overwrite == 0
    
    load(fullfile(folderName, [fileInfo.sessionName '.PBEInfo.mat']), 'PBEInfo_Bayesian', 'sdat');
    load(fullfile(folderName, [fileInfo.sessionName '.PBEInfo.mat']), 'PBEInfo')
    
else
        
    mkdir(folderName)
    
    MUA.time = []; for unit = 1:numel(spikes); MUA.time = [MUA.time; spikes(unit).time]; end
    MUA.time = sort(MUA.time, 'ascend');

    

    % detection parameters
    
    pbeDetectParams.time_resolution        = 0.001;
    pbeDetectParams.threshold              = 2;
    pbeDetectParams.exclude                = brainState.periods(ismember(brainState.periods(:, 3), [2 3]), 1:2); % limiting to NREM and QW periods
%     pbeDetectParams.exclude                = [];

    pbeDetectParams.smoothingSigma         = 0.01;
    
    pbeDetectParams.minDur                 = 0.04; % number of bins
    pbeDetectParams.maxDur                 = 0.5;
    
    pbeDetectParams.maxSilence             = 0.03;
    
    pbeDetectParams.maxVelocity            = 10; 
    pbeDetectParams.thRatioThresh          = 1;
    pbeDetectParams.rippleAThresh_low      = 1;
    pbeDetectParams.rippleAThresh_high     = 3;
    
    pbeDetectParams.binDur                 = 0.02;
    pbeDetectParams.minNTimeBins           = 4;
    pbeDetectParams.minFiringUnits         = max(5, 0.1*numel(fileInfo.okUnits)); % okUnits are all pyramidal units regardless of their stability
    

    
    % detecting primary PBEs, just the boundaries of potential PBEs before
    % applying any criteria

    [primaryPBEs, sdat] = PBPeriods(MUA, pbeDetectParams, fileInfo); % already filtering based on the duration in this step
    
    fileName = fullfile(folderName, [fileInfo.sessionName '.PBEs_before_splitting.mat']);
    save(fileName, 'primaryPBEs', 'pbeDetectParams', 'sdat', '-v7.3')
    
    
    % splitting the detected PBEs if there was a long silence in the middle
    PBEs_splitted = splitPBEs(primaryPBEs, spikes_pyr, sdat, pbeDetectParams);
    
    
    % calculate a number of features for the PBEs; brain state, number of
    % acive units, binned firing rate , etc
    
    [PBEInfo, PBEInfo_Bayesian, idx_Bayesian] = genEventStructure(PBEs_splitted, spikes_pyr, sdat, rippleLFP, pbeDetectParams, brainState, fileInfo, folderName, 'pbe');
    
end




%% calculate for each ripple concurrent brain state, binned firing rate, number of active units, etc

folderName = fullfile(storageDir,  'ripples');
fileName   = fullfile(folderName, [fileInfo.sessionName '.rippleEvents.mat']);

if ~exist(fileName, 'file') 
    
    rippleEvents = genEventStructure(ripplePeriods, spikes_pyr, sdat, rippleLFP, rplDetectParams, brainState, fileInfo, folderName, 'ripple');
    
    save(fileName, 'rippleLFP', 'rplDetectParams', 'rippleEvents', '-v7.3')
end



%% adding variables to the EEG file

nCh = fileInfo.nCh;


EEGdata = readmulti([baseFile '.eeg'], nCh, 1); % load .eeg trace
totalT  = length(EEGdata)/fileInfo.lfpSampleRate;


samplingDur  = 1/fileInfo.lfpSampleRate;
samplingPnts = samplingDur:samplingDur:totalT;

sdat_mua_2add = interp1((1:length(sdat))*1e-3, sdat, samplingPnts);
speed_2add    = interp1(fileInfo.speed.t, fileInfo.speed.v, samplingPnts);
ripple_2add   = interp1(t_1ms, power_summed, samplingPnts);
theratio_2add = interp1(timePnts, theratio, samplingPnts);


sdat_mua_2add(isnan(sdat_mua_2add)) = 0;
sdat_mua_2add(sdat_mua_2add < 0) = 0;
sdat_mua_2add(sdat_mua_2add > 3) = 3;
sdat_mua_2add = (2*(sdat_mua_2add-min(sdat_mua_2add))/(max(sdat_mua_2add)-min(sdat_mua_2add))-1)*2^10;



speed_2add(speed_2add < 10) = 0;
speed_2add = (2*(speed_2add-min(speed_2add))/(max(speed_2add)-min(speed_2add))-1)*2^11;


ripple_2add(isnan(ripple_2add)) = 0;
ripple_2add(ripple_2add > 3) = 3;
ripple_2add(ripple_2add < 0) = 0;
ripple_2add = (2*(ripple_2add - min(ripple_2add))/(max(ripple_2add)-min(ripple_2add))-1)*2^10;


theratio_2add(isnan(theratio_2add)) = 0;
theratio_2add(theratio_2add < 1) = 0;
theratio_2add = (2*(theratio_2add - min(theratio_2add))/(max(theratio_2add)-min(theratio_2add))-1)*2^11;



inputFileHandle  = fopen([baseFile,'.eeg']);
outputFileHandle = fopen([baseFile,'-2.eeg'],'w');
doneFrame = 0;
bufferSize = 4096;
while ~feof(inputFileHandle)
    data = (fread(inputFileHandle,[nCh,bufferSize],'int16'))';
    
    for frame=1:size(data,1)
        fwrite(outputFileHandle,[data(frame,:), sdat_mua_2add(frame+doneFrame), speed_2add(frame+doneFrame), ripple_2add(frame+doneFrame), theratio_2add(frame+doneFrame)]','int16'); 
    end
    doneFrame=doneFrame+frame;

end

fclose(inputFileHandle);
fclose(outputFileHandle);



%% Bayesian decoding


% BayesianReplayDetection_GrosmarkReclu_nov2020




end



%% functions

function speed = calspeed(fileInfo, position_samplingRate)

xpos = fileInfo.xyt(:, 1);
ypos = fileInfo.xyt(:, 2);

timepnts = fileInfo.xyt(:, 3);


diffx = [0; abs(diff(xpos))];
diffy = [0; abs(diff(ypos))];

difft = [1; diff(timepnts)];


velocity = sqrt(diffx.^2 + diffy.^2)./difft; 

% interpolate the velocty at time point with missing position data
velocity(isnan(velocity)) = interp1(timepnts(~isnan(velocity)), velocity(~isnan(velocity)), timepnts(isnan(velocity)));


% smoothing the speed
sigma = 0.25/(1/position_samplingRate); 
halfwidth = 3 * sigma;


smoothwin = gausswindow(sigma, halfwidth); % smoothing kernel
speed.v = conv(velocity, smoothwin, 'same'); 

% speed.v = velocity;
speed.t = timepnts;

end

