function [PBEs, sdat] = PBPeriods(spikes, detectParams, fileInfo)

% Function PBPeriods characterizes periods which later will be used for determining
% population burst events (PBEs). To do this, first, spike density function
% (sdf or called spike histogram) is calculated by counting the multiunit spikes from all units (pyramidal
% and maybe interneurons). Second, periods with sdf above x standard deviation
% above the mean in the whole period (with excluding theta or run periods)
% are characterized (pramary periods). Finally, the boundaries of the
% events will be the mean/zero-crossing points in either sides of the priamry
% periods (more specifically, the peaks within the primary periods are calculated
% and the mean-crossing points are dearched within a window around the peak)


binDur  = detectParams.time_resolution;
threshZ = detectParams.threshold;
exclude = detectParams.exclude;


% dataset info

tbegin = fileInfo.tbegin;
tend   = fileInfo.tend;

% Let's change the zero time and convert anything to second to make the next calculations easier


if isfield(spikes, 'quality') 
    
    spikeTimes = [];
    for unit = 1:numel(spikes)
        spikeTimes = [spikeTimes; spikes(unit).time];
    end

    spikeTimes = sort(spikeTimes, 'ascend');
    
else % if we use all MUA spikes
    spikeTimes = spikes.time;
end
    
spikeTimes = spikeTimes(spikeTimes > tbegin & spikeTimes < tend);
spikeTimes = spikeTimes - tbegin;



%% Calculation of Spike Density function 

% Time binnig of the spike train, bin size = 1 ms and calculating the
% firing rate

nBins = floor(spikeTimes(end) / binDur);
binEdges = 0:binDur:spikeTimes(end);

spikeBinned = histc(spikeTimes, binEdges);
spikeBinned(end) = []; %% removing the spikes that matches the last edge (we need the counts between the edges)

spikeDensity = spikeBinned/binDur;

% Final spike density function after smoothing the binned spikes


sigma     = detectParams.smoothingSigma * 1e3; %% in milisecond since we are using 1 milisecond time bins
halfwidth = 3 * sigma ;

smoothwin    = gausswindow(sigma, halfwidth);
spikeDensity = conv(spikeDensity, smoothwin, 'same');



% exclude the REM/Active wake periods if we want to limit the z-scoring to
% the NREM/quite wake periods

ii_ok = ones(length(spikeDensity),1);

if ~isempty(exclude)
    exclude = exclude((exclude(:, 2) > fileInfo.tbegin & exclude(:, 1) < fileInfo.tend), :);
end



if ~isempty(exclude)

    exclude(exclude(:,1) <= tbegin, 1) = tbegin + 0.001 * fileInfo.Fs;
    exclude(exclude(:, 2) > tend, 2)   = tend;

    exclude = exclude - tbegin; 

    exclude = floor(exclude *  1000); % in ms

    for ii = 1:size(exclude, 1)

        try
            ii_ok(exclude(ii,1):exclude(ii,2)) = 0;
        catch
            exclude(ii,1)
            exclude(ii,2)
        end
    end
end


%%%% Mean and Standard deviation of the Spike Density in the non_theta periods

rateMean = mean(spikeDensity(ii_ok>0)); 
rateSD = std(spikeDensity(ii_ok>0)); 

sdat = (spikeDensity - rateMean)/rateSD;



% PRIMARY PERIODS (periods with MUA above x SD)

%%% My strategy here was to find the periods with rate above the threshold
%%% and then find the maximum rate within those periods as the center for
%%% the rest of the analysis (e.g., defining the boundaries and so on)


% The epochs

highRateBins = find(sdat > threshZ); 

primPeriodStr = [highRateBins(1); highRateBins([0; diff(highRateBins)] > 1)];
primPeriodEnd = [highRateBins(find([0; diff(highRateBins)] > 1)-1); highRateBins(end)];


% Find the CENTER (bin with peak rate) in each of the epoches

noPrimPeriods = length(primPeriodStr);
primPeriodCntr = zeros(noPrimPeriods, 1); % the center or bin with peak rate

primpeak = zeros(noPrimPeriods, 1);

for ii = 1 : noPrimPeriods
    
    [primpeak(ii), Ind] = max(sdat(primPeriodStr(ii) : primPeriodEnd(ii))); %% Ind signifiy the center bin in respect to the current period
    primPeriodCntr(ii) = Ind + primPeriodStr(ii) - 1; 
end

primPeriods = [primPeriodStr primPeriodEnd primPeriodCntr primpeak];




% SECONDARY PERIODS (detect edges around the peak MUAs)

% Having determined the peaks, calculate the events RISING and FALLING edges 
% as first mean-passing points immediately before and following the peak. If niether found in either side within a
% search window, that event will be omitted.


halfSearchLen = 1000; % in ms the search period from either sides of the peaks for mean-crossing point.

% Initializing

risingEdge = zeros(noPrimPeriods, 1);
fallingEdge = zeros(noPrimPeriods, 1);
eventCntr = zeros(noPrimPeriods, 1);
eventPeakFiring = zeros(noPrimPeriods, 1);

% Lets do the calculations for all the PRIMARY periods and correspnding
% peaks

epochInd = 0; % indices of the epochs with side mean-passing points within search windows of 1000 ms toward each side of the peak 

for epoch = 1 : noPrimPeriods
    
    
    peakTime = primPeriods(epoch, 3); % peak time of the current event
    
    startsearchPnt = max(1, peakTime-halfSearchLen); 
    endsearchPnt   = min(nBins, peakTime+halfSearchLen);
    
    
    
    % Find the mean-passing points prior to the peak (RISING EDGE)
    
    zeroCrossing_prior = find(sdat(startsearchPnt:peakTime) < 0, 1, 'last');
    
    
    
    % Find the mean-passing points post to the peak (FALLING EDGE)
    
    zeroCrossing_post = find(sdat(peakTime : endsearchPnt) < 0, 1, 'first');
        
    if isempty(zeroCrossing_prior) || isempty(zeroCrossing_post) || startsearchPnt == 1 || endsearchPnt == nBins
       continue
    else
        
       epochInd = epochInd + 1;
       
       risingEdge(epochInd) = peakTime - (halfSearchLen-zeroCrossing_prior)-1;
       fallingEdge(epochInd) = peakTime + zeroCrossing_post - 1;
       
       eventCntr(epochInd) = peakTime;
       eventPeakFiring(epochInd) = primPeriods(epoch, 4);
       
    end
    
end



% Create the secondary periods

secondaryPeriods = [risingEdge fallingEdge eventCntr eventPeakFiring]; %% the third and fourth columns are the center and the peak
secondaryPeriods(risingEdge == 0, :) = []; %% removing the extra zeros



% get rid of repeated PBEs
min_distance = 0;
tmp_pbe = secondaryPeriods(1,:);
third = [];

for ii = 2:size(secondaryPeriods,1)
    if (secondaryPeriods(ii,1)-tmp_pbe(2)) < min_distance
        
        if secondaryPeriods(ii,4) > tmp_pbe(4)
            currpeakT   = secondaryPeriods(ii, 3);
            currpeakMUA = secondaryPeriods(ii, 4);
        else
            currpeakT   = tmp_pbe(3);
            currpeakMUA = tmp_pbe(4);
        end
        
        tmp_pbe = [tmp_pbe(1) max(tmp_pbe(2), secondaryPeriods(ii,2)) currpeakT currpeakMUA]; % merge two PBEs
    else
        third = [third; tmp_pbe];
        tmp_pbe = secondaryPeriods(ii,:);
    end
end

third = [third;tmp_pbe]; 


PBEs = third(:, 1:3)*binDur + fileInfo.tbegin; %%% the fourth column is just the peak rate
PBEs(:,4) = third(:, 4);


% filtering based on the duration
duration = PBEs(:, 2) - PBEs(:, 1);
PBEs = PBEs(duration >= detectParams.minDur & duration <= detectParams.maxDur, :);



% store the PBEs in a structure
PBEs2 = struct('startT', [], 'endT', [], 'peakT', [], 'peakMUA', [], 'n', []);
for ipbe = 1:size(PBEs, 1)
   PBEs2(ipbe).startT  = PBEs(ipbe, 1);
   PBEs2(ipbe).endT    = PBEs(ipbe, 2);
   PBEs2(ipbe).peakT   = PBEs(ipbe, 3);
   PBEs2(ipbe).peakMUA = PBEs(ipbe, 4);
   PBEs2(ipbe).n       = ipbe; % a number to track PBEs over the next splitting and qualification processes
end
PBEs = PBEs2;



end

