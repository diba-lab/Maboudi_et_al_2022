function [spikes, template, linearPoscenters] = spatialTuning_1D_June21(spikes, behavior, thetaPeriods, turningPeriods, speed, runSpeedThresh, direction, posBinSize, fileInfo, fileBase)



% including only the spikes meeting certain criteria

if strcmp(direction, 'LR')
    desiredMod = 0; % even laps
    directionNum = 1;
    unitSortingMode = 'ascend';
elseif strcmp(direction, 'RL')
    desiredMod = 1; % odd laps
    directionNum = 2;
    unitSortingMode = 'descend';
elseif strcmp(direction, 'uni')
    desiredMod = [0 1];
    directionNum = 1;
    unitSortingMode = 'ascend';
end


% finding position samples pertaining to the maze period and certain direction of travel 

speedatPos  = interp1(speed.t, speed.v, fileInfo.xyt(:, 3));

positionIdx = find(fileInfo.xyt(:, 3) > behavior.time(4,1) & fileInfo.xyt(:, 3) < behavior.time(4,2) ... % within the run period
                    & speedatPos > runSpeedThresh ... % position bins when animal's velocity is higher than threshold
                    & fileInfo.linearPos(:, 3) > 0 & ismember(mod(fileInfo.linearPos(:, 3), 2), desiredMod)); % limiting to the travels in specific direction

                
allLaps = unique(fileInfo.linearPos(:, 3));
currDirLaps = allLaps(ismember(mod(allLaps, 2), desiredMod));
nLaps = numel(currDirLaps);



% include only the position samples within the theta peridos
if ~isempty(thetaPeriods)
    thetaPositionInd = []; 
    for ii = 1: size(thetaPeriods, 1) 
        thetaPositionInd = [thetaPositionInd; find(fileInfo.xyt(:, 3) >= thetaPeriods(ii, 1) & fileInfo.xyt(:, 3) <= thetaPeriods(ii, 2))];
    end     
    positionIdx = intersect(positionIdx, thetaPositionInd);
end




% exclude the position samples within the turning periods 
if ~isempty(turningPeriods)
    turningPositionInd = []; % should be excluded
    for ii = 1: size(turningPeriods, 1)
        turningPositionInd = [turningPositionInd; find(fileInfo.xyt(:, 3) >= turningPeriods(ii, 1) & fileInfo.xyt(:, 3) <= turningPeriods(ii, 2))]; 
    end     
    positionIdx = setdiff(positionIdx, turningPositionInd);
end


linearPos = fileInfo.linearPos(:, [1 3]); % the third row indicates lap indices of the positions                   
directionLinearPos = nan(size(linearPos));          
directionLinearPos(positionIdx, :) = linearPos(positionIdx, :);




% defining the position bins

nPosBins = floor((max(linearPos(:, 1)) - min(linearPos(:, 1)))/posBinSize);
posBinEdges = min(linearPos(:, 1)): posBinSize: max(linearPos(:, 1)); % center of the position bins
linearPoscenters = posBinEdges(1:end-1) + posBinSize/2;
posSamplingPeriod = median(diff(fileInfo.xyt(:, 3))); % 1/sampling frequency - Hiro's dataset: timeunit is microsecond


nUnits = numel(spikes);
spatialInfo_all   = zeros(nUnits, 1);
conslapsratio_all = zeros(nUnits, 1);
peakFR_all        = zeros(nUnits, 1);
peakPosBin_all    = zeros(nUnits, 1);

diffWithAvg   = zeros(numel(currDirLaps), nUnits);

for unit = 1:nUnits
    
    spikeInd = find(spikes(unit).time >= behavior.time(4,1) & spikes(unit).time < behavior.time(4,2) ... % within the RUN period
                    & spikes(unit).speed > runSpeedThresh ... % only the spikes happening when the velocity of the animal is higher than the threshold (usually 10 cm/sec)
                    & spikes(unit).lap' > 0 & ismember(mod(spikes(unit).lap', 2), desiredMod));  % only spike during high theta power
    %                 & spikeStruct.theta == 1); % limiting to the travels in specific direction

    % excluding the spikes that occurr when the animal stopped or turned
    % around in the middle of the track

    turningSpikes = [];
    for ii = 1: size(turningPeriods, 1)
        turningSpikes = [turningSpikes; find(spikes(unit).time >= turningPeriods(ii, 1) & spikes(unit).time <= turningPeriods(ii, 2))]; 
    end

    
    spikeInd = setdiff(spikeInd, turningSpikes);

    
    spikePositions = spikes(unit).linearPos(spikeInd);
    spikeLap       = spikes(unit).lap(spikeInd); 
    
    posBinnSpikeCnts_laps = zeros(nPosBins, nLaps);
    posBinDwelltime_laps  = zeros(nPosBins, nLaps);
        
    placeFields_laps = zeros(nPosBins, nLaps);
    peakPosBin_laps  = zeros(nLaps, 1);

    for jj = 1:nLaps
        
        positions = directionLinearPos(directionLinearPos(:, 2) == currDirLaps(jj), 1);

        unitSpikePositions = spikePositions(spikeLap == currDirLaps(jj));
        
        if isempty(unitSpikePositions)
            continue
        end
        
        temp   = histc(unitSpikePositions, posBinEdges);
        posBinnSpikeCnts_laps(:, jj) = temp(1:end-1);


        temp    = histc(positions, posBinEdges) * posSamplingPeriod;
        posBinDwelltime_laps(:, jj)  = temp(1:end-1);
        
        
        unsmoothed_tunings = posBinnSpikeCnts_laps(:, jj) ./ posBinDwelltime_laps(:, jj);

        unsmoothed_tunings(isnan(unsmoothed_tunings)) = 0;
        unsmoothed_tunings(isinf(unsmoothed_tunings)) = 0;


        win = gausswindow(3,5); % smoothing with a standard deviation of 6 cm (assuming that each positon bins is 2cm long)

        placeFields_laps(:, jj) = conv(unsmoothed_tunings, win, 'same');
        
        if sum(placeFields_laps(:, jj)) > 0
            [~, peakPosBin_laps(jj)] = max(placeFields_laps(:, jj));
        else
            peakPosBin_laps(jj) = nan;
        end
    
    end
    
    
    % average across the laps 
    posBinnSpikeCnts = sum(posBinnSpikeCnts_laps, 2);
    posBinDwelltime  = sum(posBinDwelltime_laps, 2);
    
    unsmoothed_tunings = posBinnSpikeCnts./posBinDwelltime;

    unsmoothed_tunings(isnan(unsmoothed_tunings)) = 0;
    unsmoothed_tunings(isinf(unsmoothed_tunings)) = 0;
    
    
    spikes(unit).spatialTuning_unsmoothed_re.(direction) = unsmoothed_tunings; 
    
    
    win = gausswindow(3,5);
    spikes(unit).spatialTuning_smoothed_re.(direction) = conv(unsmoothed_tunings, win, 'same');
    [spikes(unit).peakFR_re.(direction), spikes(unit).peakPosBin_re.(direction)] = max(spikes(unit).spatialTuning_smoothed_re.(direction));
    
    
    % investigate consistence of firings across the laps
    tolerationLimit = 30; % in cm
    tolerationLimit = floor(tolerationLimit/posBinSize/2);
    
    diffWithAvg(:, unit) = peakPosBin_laps - spikes(unit).peakPosBin_re.(direction);
    consLaps = find(abs(diffWithAvg(:, unit)) < tolerationLimit); % laps with consistent peak position for each unit
    spikes(unit).conslapsRatio_re.(direction) = numel(consLaps)/nLaps;
    
    
    % spatial inforamtion
    pi = posBinDwelltime/sum(posBinDwelltime);
    tempPlaceMap = spikes(unit).spatialTuning_smoothed.(direction) + 1e-4; % add a small value to remove zero firing rates
    
    FR = sum(tempPlaceMap .* pi); % average firing rate
    spikes(unit).spatialInfo_re.(direction) = sum((tempPlaceMap/FR .* log2(tempPlaceMap/FR)) .* pi);
    
    
    spatialInfo_all(unit)   = spikes(unit).spatialInfo_re.(direction);
    conslapsratio_all(unit) = spikes(unit).conslapsRatio_re.(direction);
    peakFR_all(unit)        = spikes(unit).peakFR_re.(direction);
    peakPosBin_all(unit)    = spikes(unit).peakPosBin_re.(direction);
 

end



placeCells = find(spatialInfo_all > 0 & conslapsratio_all > 0 & peakFR_all > 1); % here we are not using the consistence and spatial info to filter cells


peakPosBin_all = peakPosBin_all(placeCells);
[~, sortIdx]   = sort(peakPosBin_all, unitSortingMode);
template       = placeCells(sortIdx);




% plot the place fields

figure;

x0=0;
y0=0;
width=250;
height=100*numel(placeCells);
difpos = linearPoscenters(2)- linearPoscenters(1);
set(gcf,'units','points','position',[x0,y0,width,height])


for pc = 1:numel(placeCells)
    
    unit = template(pc); 

    PF = spikes(unit).spatialTuning_smoothed_re.(direction);
    PF_normalized = PF'./max(PF);
    
    fill([linearPoscenters fliplr(linearPoscenters)], [0.06*pc + PF_normalized/20 fliplr(0.06*pc*ones(size(PF_normalized)))], 'r','LineStyle','none')
    hold on

    plot(linearPoscenters, 0.06*pc + PF_normalized/20,'color', 'k','linewidth', 0.5);
    alpha(0.5)
    set(gca, 'YTick', [], 'YTickLabel', [], 'color', 'none', 'YColor', 'none', 'box', 'off', 'fontsize', 12)

end


xlim([linearPoscenters(1)-difpos/2 linearPoscenters(end)+difpos/2])

xlabel('Position on track(cm)', 'fontsize', 14)

h = text(linearPoscenters(1)-5*difpos, 0.06*numel(placeCells)/2, 'Units', 'fontsize', 14, 'HorizontalAlignment', 'center');
set(h, 'rotation', 90)



if ~strcmp(direction, 'uni')

    ha = annotation('arrow');  
    ha.Parent = gca;  

    if strcmp(direction, 'LR')
        ha.X = [linearPoscenters(1) linearPoscenters(15)]; 
    elseif strcmp(direction, 'RL')
        ha.X = [linearPoscenters(end) linearPoscenters(end-15)]; 
    end

    ha.Y = [0.06*(numel(placeCells)+2) 0.06*(numel(placeCells)+2)];   

    ha.LineWidth  = 2;          % make the arrow bolder for the picture
    ha.HeadWidth  = 10;
    ha.HeadLength = 10;
end



if ~isempty(fileBase)

    if ~strcmp(direction, 'uni')

        filename = [fileInfo.sessionName '_placeFields1D_' direction];

    else
        filename = [fileInfo.sessionName '_placeFields1D_uni'];
    end


    savepdf(gcf, fullfile(fileBase, filename), '-dpng')
    savefig(gcf, fullfile(fileBase, filename))
end



%%% making clu and res files 

rest = [];
cluorder = [];

for pc = 1 : numel(placeCells)
    
    unit = template(pc);

    spikeTimes = floor(spikes(unit).time * fileInfo.Fs)';
    rest = [rest spikeTimes];
    cluorder = [cluorder; pc*ones(length(spikeTimes),1)];

end

[sortedRest, ind] = sort(rest);
cluorder = cluorder(ind)-1;


Saveres([fileBase '/' fileInfo.sessionName direction 'Active.10' num2str(directionNum) '.res.' num2str(directionNum)],sortedRest);
SaveClu([fileBase '/' fileInfo.sessionName direction 'Active.10' num2str(directionNum) '.clu.' num2str(directionNum)],cluorder);



end