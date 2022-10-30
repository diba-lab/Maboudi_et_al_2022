function postPr = BayesianDecoder(binnedFiring, tuning, binDur)


% The difference with the version above is that we calculate the log
% probabilities (logProbs) and at the end with revert it back to praobility values by
% calcualting exponentials og the logProb values

% logProb = -\tau*(\lambda1 + \lambda2 + ... + \lambdaN) + n1*log(\tau * \lambda1) + n2*log(\tau * \lambda2) + ... + nN*log(\tau * \lambdaN)
% logProb = firstTerm + secondterm;
% prob    = exp(logProb);


noPosBins = size(tuning, 2); 
nTimeBins = size(binnedFiring, 2);


logTunings      = log(binDur * tuning);
tuningSummation = - binDur * sum(tuning, 1);



if nTimeBins < 1000 % this is suitbale if we are decoding bins within individual PBEs

    logPr = sum(repmat(permute(logTunings, [1 3 2]), [1 nTimeBins 1]) .* repmat(binnedFiring, [1 1 noPosBins]), 1) + ...
         repmat(permute(tuningSummation, [1 3 2]), [1 nTimeBins 1]);

    logPr  = permute(logPr, [3 2 1]);
    postPr = exp(logPr);
    
else
    
    maxBins   = 1000;
    nSegments = ceil(nTimeBins/maxBins);
    
    postPr = nan(noPosBins, nTimeBins);
    
    for ii = 1:nSegments
        
        currTimeBins  = (ii- 1)*maxBins+1 : min(ii* maxBins, nTimeBins); 
        nCurrTimeBins = numel(currTimeBins);
        
        logPr = sum(repmat(permute(logTunings, [1 3 2]), [1 nCurrTimeBins 1]) .* repmat(binnedFiring(:, currTimeBins), [1 1 noPosBins]), 1) + ...
         repmat(permute(tuningSummation, [1 3 2]), [1 nCurrTimeBins 1]);

        logPr  = permute(logPr, [3 2 1]);
        postPr(:, currTimeBins) = exp(logPr);
        
    end
    
end


end

