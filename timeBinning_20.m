function [binnedFiring, binCenters] = timeBinning_20(epoch, spikes, binDur, binOverlapRatio, fileInfo)

% This function is intended to bin the spikes trains of pyramidal units
% (idnetifies by qclus) within each event. 

% INPUTS
% binOverlapRatio: the overlap between the successive bins

overlap  = binOverlapRatio * binDur;
stepSize = binDur - overlap; 


startT = epoch(1); 
endT   = epoch(2); 


nUnits = numel(spikes); % total number of units

% okUnits  = fileInfo.stablePyr;
okUnits  = fileInfo.okUnits;
nOKUnits = numel(okUnits); % total number of units which are pyramidal and stable


binStarts  = epoch(1):stepSize:(epoch(2)- binDur);% for coarse time bins in ms
binCenters = binStarts + binDur/2;
nBins      = length(binStarts);


binnedFiring = zeros(nUnits, nBins);

for iUnit = 1:nOKUnits

    unit = okUnits(iUnit);

    spikeTimes = spikes(unit).time;
    spikeTimes = spikeTimes(spikeTimes >= startT & spikeTimes < endT);


    spikeTrainSegSize = 100; % Due to the long recording, the spike trains were divided into mutiple segments and a summation is calaculated at the end

    nSegments = ceil(numel(spikeTimes)/spikeTrainSegSize);


    segCounts = zeros(nSegments, nBins);
    for jj = 1:nSegments 

       currentSpikeTimes = spikeTimes((jj-1)*spikeTrainSegSize+1: min(numel(spikeTimes), jj*spikeTrainSegSize));
        
       
%        firstBin = find(binStarts < currentSpikeTimes(1), 1, 'last');
%        lastBin  = find(binStarts > currentSpikeTimes(end), 1, 'first');
%        
%        currBins = firstBin:lastBin;
        
        currBins = 1:nBins;
        
    
       respSpikeTimes  = (repmat(currentSpikeTimes, [1 numel(currBins)]) - repmat(binStarts(currBins), [numel(currentSpikeTimes) 1])); % spike times in respect to start times of the bins
       ifWithinBin     = respSpikeTimes >= 0 & respSpikeTimes < binDur; % a row corresponding to each spike

       segCounts(jj, currBins) = sum(ifWithinBin, 1);

    end
    binnedFiring(unit, :) = sum(segCounts, 1); 

end


end



