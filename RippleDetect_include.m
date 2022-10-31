function [ripple_result, filtered_data, sdat, includeIdx] = RippleDetect_include(FileBase, fileinfo, detectParams)
% function [ripple_result] = RippleDetect(fileinfo,without,max_thresholdf,homefilter,runfilter,thetafilter)
%
% function is intended to detect ripple events, from 130-230 Hz, in the eeg file, given
% fileinfo, an array which contains the name and selected channels of the
% directory in question.

max_thresholdf = detectParams.threshold;
exclude        = detectParams.exclude;
period         = [];


% currentdir = pwd;
% FileBase = [currentdir '/' fileinfo.name '/' fileinfo.name];

fprintf('%s  detecting ripples...',fileinfo.sessionName)

highband = 250; % bandpass filter range
lowband = 150; 

thresholdf = 1  ; % power SD threshold for ripple detection (data mean in case of zero)

min_sw_period = 15 ; % minimum ripple period, 50ms ~ 5-10 cycles\
max_sw_period = 1000;
min_isw_period = 0; % minimum inter-ripple period



%%%%% Configuration %%%%%

Par = LoadPar([FileBase '.xml']); % either par or xml file needed

numchannel = Par.nChannels; % total channel number, 6, 10 or 18
EegRate = Par.lfpSampleRate; % sampling rate of eeg file


forder = 100;  % filter order has to be even. The longer the more selective, but the operation
% will be linearly slower to the filter order. 100-125 for 1.25Hz (500 for 5 KH)

avgfilorder = round(min_sw_period/1000*EegRate/2)*2+1 ; % should not change this. length of averaging filter
% avgfilorder = 101; % should not change this. length of averaging filter
forder = ceil(forder/2)*2; % to make sure filter order is even

select_channel = fileinfo.RippleChannels; % tetrode/silicon probe 1-4,8,16ch
 


firfiltb = fir1(forder,[lowband/EegRate*2, highband/EegRate*2]); % calculate convolution func


sigma = 10/1000*EegRate; %% in lfp sampleing period
halfwidth = 3 * sigma;
smoothwin = gausswindow(sigma, halfwidth);


if exist([FileBase '.eeg'], 'file')
   filtered_data = readmulti([FileBase '.eeg'],numchannel,select_channel); % load .eeg trace
elseif exist([FileBase '.lfp'], 'file')
   filtered_data = readmulti([FileBase '.lfp'],numchannel,select_channel); % load .eeg trace
else
    error('eeg file is missing!')
end


if ~isempty(period)
    filtered_data = filtered_data(period(1)*EegRate+1 : period(2)*EegRate, :);
end

filtered_data = Filter0(firfiltb, filtered_data); % the actual filtering

  
filtered_data2 = abs(hilbert(filtered_data)); % to power trace
filtered_data2 = conv(sum(filtered_data2,2), smoothwin, 'same');
filtered_data2 = sum(filtered_data2, 2); % taking average over the channels



filtered_data3 = nan(size(filtered_data2));

includeIdx = true(size(filtered_data3));
timePnts   = (1:length(filtered_data3))/EegRate; 


for iperiod = 1:size(exclude, 1)
    
   includeIdx(timePnts > exclude(iperiod,1) & timePnts < exclude(iperiod, 2)) = false; 
    
end


filtered_data3(includeIdx) = filtered_data2(includeIdx);
filtered_data3(filtered_data3 > prctile(filtered_data3, 99.5)) = nan;

sdat = (filtered_data2 - repmat(nanmean(filtered_data3), [size(filtered_data3, 1), 1])) ./ repmat(nanstd(filtered_data3), [size(filtered_data3, 1), 1]);


% (1) primary detection of ripple periods, based on thresholding
thresholded = sdat > thresholdf;
primary_detection_b = find([0; diff(thresholded)]>0);
primary_detection_e = find([0; diff(thresholded)]<0);

if primary_detection_b(1) > primary_detection_e(1) % in case the period of interset started in the middle of a ripple 
    primary_detection_e = primary_detection_e(2:end);
end

sum_detection =[primary_detection_b(1:size(primary_detection_e,1),1) primary_detection_e]; % in case the the period ended in the middle of a ripple
% sum_detection_e = [sum_detection_e primary_detection_e];

sum_detection = sortrows(sum_detection);
primary_detection_b = sum_detection(:,1);
primary_detection_e = sum_detection(:,2);

primary = [primary_detection_b,primary_detection_e]; % [start, end]


% (2) merging ripples, if inter-ripples period is less than min_isw_period;
min_isw_period = min_isw_period/1000*EegRate; % in eeg
min_sw_period = min_sw_period/1000*EegRate; % in eeg
% max_sw_period = max_sw_period/1000*EegRate;
secondary=[];


tmp_rip = primary(1,:);

for ii=2:size(primary,1)
    if (primary(ii,1)-tmp_rip(2)) < min_isw_period
        tmp_rip = [tmp_rip(1), max(tmp_rip(2), primary(ii,2))]; % merge two ripples
    else
        secondary = [secondary;tmp_rip];
        tmp_rip = primary(ii,:);
    end
end

secondary = [secondary;tmp_rip]; % [start, end]
secondary = secondary((secondary(:,2)-secondary(:,1)) > min_sw_period,:);


% (3) ripples must have its peak power of > max_thresholdf

third = zeros(size(secondary, 1), 4);
for ii=1:size(secondary,1)
    [max_val,max_idx] = max(sdat(secondary(ii,1):secondary(ii,2)));
    third(ii, :) = [secondary(ii,:) secondary(ii,1)+max_idx-1 max_val];

end

third = third(third(:,4)> max_thresholdf, :);



% (7) save detected ripples
ripple_result = third; % [start, peak, end, SD] (in eeg)
ripple_result(:, 1:3) = ripple_result(:, 1:3)/EegRate;



ripple_result2 = struct('startT', [], 'endT', [], 'peakT', [], 'peakRippleA', []);
for rpl = 1:size(ripple_result, 1)
    ripple_result2(rpl).startT      = ripple_result(rpl, 1);
    ripple_result2(rpl).endT        = ripple_result(rpl, 2);
    ripple_result2(rpl).peakT       = ripple_result(rpl, 3);
    ripple_result2(rpl).peakRippleA = ripple_result(rpl, 4);
end
ripple_result = ripple_result2;


end


