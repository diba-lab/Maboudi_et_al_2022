function [replayScore, weightedCorr, jumpDistance, onlyLineElements, begPnt, endPnt] = calReplayMetrics(postPr, varargin)


% weighted correlation 
weightedCorr = calWeightedCorr(postPr);


% find the best fit line



[nPosBins, nTimeBins] = size(postPr);

% replacing each element within a column with a summation over the neighboring elements
% (corresponding to 30 sm vicinity (-15 cm to 15 cm))


%%% I can do this once and outside the function. it
%%% helps to save time for the shuffles

nVicinityBins = 8; 
postPr_summed = zeros(size(postPr));

for ii = 1: nTimeBins
    postPr_summed(:, ii) = conv(postPr(:, ii), ones(2*nVicinityBins+1, 1), 'same');
end


% lookupIdx = cell2mat(lineDictionary(:, 3)) == nTimeBins;

if nargin > 1
   currlineDictionary = varargin{1};
else
   currlineDictionary = generateLineDictionary(nPosBins, nTimeBins);
end

% currlineDictionary = lineDictionary(lookupIdx, :);

% clear lineDictionary

nLines  =  size(currlineDictionary{1}, 3);


postPrMultipliedWithLines = repmat(postPr_summed, [1 1 nLines]) .* currlineDictionary{1};


elementsOnLines = permute(sum(postPrMultipliedWithLines), [3 2 1]);
medianMatrix    = repmat((median(postPr_summed)), [nLines, 1]);



elementsOnLines(isnan(elementsOnLines)) = medianMatrix(isnan(elementsOnLines));

sumProb = sum(elementsOnLines, 2)/nTimeBins;

[replayScore, lindIdx] = max(sumProb(:));

bestTheta  = currlineDictionary{2}(lindIdx, 1);
bestLength = currlineDictionary{2}(lindIdx, 2);

onlyLineElements = postPr_summed .* currlineDictionary{1}(:, :, lindIdx);

%%% start and end points of the best fitting line

t_mid = (nTimeBins - 1)/2;
p_mid = (nPosBins - 1)/2;

tBeg = 1;
yBeg = floor((bestLength - (tBeg-t_mid)*cos(bestTheta))/sin(bestTheta) + p_mid);

begPnt = [tBeg yBeg];



tEnd = nTimeBins;
yEnd = floor((bestLength - (nTimeBins-t_mid)*cos(bestTheta))/sin(bestTheta) + p_mid);

endPnt = [tEnd yEnd];




% slope = tan(bestTheta);
% rho   =  yBeg ;

% 
% 
% figure; imagesc(postPr); colormap('jet'); set(gca, 'yDir', 'normal')
% hold on
% line([begPnt(1) endPnt(1)], [begPnt(2) endPnt(2)], 'linewidth', 3, 'color', 'w')
% hold on
% text(t_mid, p_mid, '*', 'fontsize', 10, 'color', 'w')
% 
% figure; imagesc(sumProb); colormap('jet'); set(gca, 'yDir', 'normal')

%% jump distance
% tic

[~, peakPos] = max(postPr, [], 1);


% eliminating non-firing time bins
sumProb                = sum(postPr, 1);
peakPos(sumProb < 0.1) = [];


% calculating jump distances



if numel(peakPos) < 2
   
   maxJump     = 1;
   medianJump  = 1;
   normMaxJump = 1;
   
else

    jumpDistances = abs(diff(peakPos));
    
    maxJump       = max(jumpDistances)/nPosBins; % maximum jump distance as a ratio of the track length
    medianJump    = median(jumpDistances)/nPosBins; % median jump distance as a ratio of the track length


    % normalized maximum jump distance (data as a percentile of shuffle distribution:within-PBE time bin shuffle (ref: Dragoi2019 Science))
    % compensating for the length of the event
    
    nShuffles      = 500; 
    shuffleMaxJump = zeros(nShuffles, 1);

    for ii = 1: nShuffles

        shufflePeakPos      = peakPos(randperm(length(peakPos)));
        shuffleJumpDistance = abs(diff(shufflePeakPos));


        shuffleMaxJump(ii)  = max(shuffleJumpDistance)/nPosBins;

    end
    normMaxJump  = length(find(shuffleMaxJump < maxJump))/nShuffles;

end


jumpDistance = [maxJump normMaxJump medianJump];



end