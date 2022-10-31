function PBEs_splitted = splitPBEs(PBEs, spikes_pyr, sdat, detectParams)
    
    spikes_pyr_pooled = cell(numel(spikes_pyr), 1);
    for iUnit = 1:numel(spikes_pyr)
        spikes_pyr_pooled{iUnit} = spikes_pyr(iUnit).time';
    end
    spikes_pyr_pooled = cell2mat(spikes_pyr_pooled');
    spikes_pyr_pooled = sort(spikes_pyr_pooled, 'ascend');



    PBEs_splitted = struct('startT', [], 'endT', [], 'peakT', [], 'peakMUA', [], 'n', []);
    tt = 1;
    for ipbe = 1:numel(PBEs)

        currSpikes = spikes_pyr_pooled(spikes_pyr_pooled >= PBEs(ipbe).startT & spikes_pyr_pooled <= PBEs(ipbe).endT);
        cutIdx     = find([0 diff(currSpikes)] > detectParams.maxSilence); % indices of the spikes that occurred longer than 30 ms after the previous spike

        if isempty(cutIdx)
            % the original PBE is passed
            
            if isempty(currSpikes)  
                PBEs_splitted(tt).startT  = PBEs(ipbe).startT;
                PBEs_splitted(tt).endT    = PBEs(ipbe).endT;
            else
                PBEs_splitted(tt).startT  = currSpikes(1)-0.015;
                PBEs_splitted(tt).endT    = currSpikes(end)+0.015;
            end
            
            PBEs_splitted(tt).peakT   = PBEs(ipbe).peakT;
            PBEs_splitted(tt).peakMUA = PBEs(ipbe).peakMUA;
            PBEs_splitted(tt).n       = PBEs(ipbe).n;

            tt = tt + 1;
        else
            % the PBE is split to multiple shorter PBEs
            nseg = numel(cutIdx)+1;
                
            split_starts = [currSpikes(1)-0.01 currSpikes(cutIdx)-0.01];
            split_ends   = [currSpikes(cutIdx-1)+0.01 currSpikes(end)+0.01];

            for isplit = 1:nseg

                [max_MUA, peakIdx] = max(sdat((floor(split_starts(isplit)*1e3) : floor(split_ends(isplit)*1e3))));

                duration =  split_ends(isplit)- split_starts(isplit);
                if duration >= detectParams.minDur
                    PBEs_splitted(tt).startT  = split_starts(isplit);
                    PBEs_splitted(tt).endT    = split_ends(isplit);
                    PBEs_splitted(tt).peakT   = split_starts(isplit)+peakIdx*1e-3;
                    PBEs_splitted(tt).peakMUA = max_MUA;
                    PBEs_splitted(tt).n       = PBEs(ipbe).n;

                    tt = tt + 1;
                end

            end
        end
    end
end