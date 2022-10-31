function  clusterQuality = calClusterQuality_2(baseFile)

% This function calculates the degree of overlap (based on Mahalanobis distance) between
% the clusters (cloud) of spikes correponding to different putiative neurons in a multidimensional feature space
% The feature space consisits of the spike amplitudes on different
% recroding channels (up to 128)

[filePath, name] = fileparts(baseFile);

fileNameBase = fullfile(filePath, name);

try
    Par = LoadXml([fileNameBase '.xml']);
catch
    Par = LoadXml(fileNameBase);
end


if isfield(Par, 'SpkGrps')
    shankInfo = Par.SpkGrps;
elseif isfield(Par, 'ElecGp')
    shankInfo = Par.ElecGp;
elseif isfield(Par, 'AnatGrps')
    shankInfo = Par.AnatGrps;
else
    error('\nShank info was not found!')
end    
nShanks = numel(shankInfo);



clusterQuality = struct('mahalDist', [], 'isoDist', [], 'Lratio', [], 'clus', []);

for iShank = 1:nShanks
    
    if isfield(shankInfo, 'Skip')
        channs = shankInfo(iShank).Channels(~shankInfo(iShank).Skip);
    else
        channs = shankInfo(iShank).Channels;
    end

    
    nChannels = numel(channs);    
  
    
    spkFileName = [fileNameBase '.spk.' num2str(iShank)];
    
    if exist(spkFileName, 'file')
        [spikeWaveForms, nChannels] = readSpk(spkFileName, nChannels, 32); % spikeWaveForms = [nChannels * nSamplesPerSpike * nSpikes]
    else
        continue
    end
    
    ifanyAmp = squeeze(sum(sum(spikeWaveForms, 1), 2));
    spikeWaveForms = spikeWaveForms(:, :, ifanyAmp~=0);
    
    
    nSpikes = size(spikeWaveForms, 3);
  

    % calculating the PCs, power and the range of the 
    
    PCAs_global = zeros(3, nChannels, nSpikes);
    for k = 1:nChannels 
         [~, scores] = pca(permute(spikeWaveForms(k,:,:),[3,2,1]),'NumComponents',3); % do we need the z-scoring here before PCA??
         PCAs_global(:,k,:) = scores';
    end
    featureFile2 = reshape(PCAs_global, [nChannels*3, nSpikes]);
    
    cluFile = load([fileNameBase '.clu.' num2str(iShank)]);
    
    nClus = cluFile(1);
    cluFile = cluFile(2:end);
    cluFile = cluFile(ifanyAmp ~= 0);
    uniqClus = unique(cluFile);
    

    cluFeatures   = cell(nClus, 1);
    spkWv_mean    = cell(nClus, 1);
    spkWv_peakAmp = cell(nClus, 1);
    
    nClusterSpikes = zeros(nClus, 1);
    for iClu = 1:nClus
        currClu = uniqClus(iClu);
        cluFeatures{iClu}    = featureFile2(:, cluFile == currClu);
        nClusterSpikes(iClu) = numel(find(cluFile == currClu));
        
        spkWv_mean{iClu}    = nanmean(spikeWaveForms(:,:, cluFile == currClu), 3);
        spkWv_peakAmp{iClu} = min(spkWv_mean{iClu}, [], 2);
        
    end
    
    clear spikeWaveForms

    clusterQuality(iShank).Lratio  = nan(nClus, nClus);
    
    
    for iClu = 1:nClus
        
       for jClu = setdiff(1:nClus, iClu)
           
           
           % find up to 4 channels with maximum difference between the two
           % units
           
           nSelectChann = 4;
           
           spkAmpDiffs = abs(spkWv_peakAmp{jClu} - spkWv_peakAmp{iClu});
           [~, sortIdx] = sort(spkAmpDiffs, 'descend');
           channsWithMaxDiffs = sortIdx(1:nSelectChann);
           
           
           featuresIdx2Include = nan(nSelectChann*3, 1);
           for ch = 1:nSelectChann
               firstFeature = (channsWithMaxDiffs(ch)-1)*3+1;
               featuresIdx2Include((ch-1)*3+1:ch*3) = firstFeature:(firstFeature+2);
           end
           df = nSelectChann*3;
           
           refClu = cluFeatures{iClu}(featuresIdx2Include, :)';
           testClu = cluFeatures{jClu}(featuresIdx2Include, :)';
           
           
           try % in case there are not enough spikes (less than the number of features, it gives an error)
               
               allMahalDist = mahal(testClu, refClu);               
                
               L = sum(1-chi2cdf(allMahalDist, df));
               clusterQuality(iShank).Lratio(jClu, iClu) = L/nClusterSpikes(iClu);
               

           catch

               clusterQuality(iShank).Lratio(jClu, iClu)    = nan;

           end
           
       end
        
    end
    clusterQuality(iShank).clus = uniqClus;  
   
end

clusterQuality = rmfield(clusterQuality, 'mahalDist');
save(fullfile(filePath, [name '.clusterQuality.mat']), 'clusterQuality')



