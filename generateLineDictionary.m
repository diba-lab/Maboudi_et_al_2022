function lineDictionary = generateLineDictionary(nPosBins, timeBinSizes) 

nTimeBinSizes = numel(timeBinSizes);


minTheta = pi/2 - atan(nPosBins/3);

nThetas  = 100; 
nLengths = 100;
nLines   = nThetas * nLengths;


thetas   = linspace(minTheta, 2*pi-minTheta, nThetas);

lineDictionary = cell(nTimeBinSizes, 3);

for kk = 1:nTimeBinSizes 
    
nTimeBins = timeBinSizes(kk);


t_mid = (nTimeBins - 1)/2;
p_mid = (nPosBins - 1)/2;


diam = sqrt((nTimeBins-1)^2 + (nPosBins-1)^2);

maxLength = max(nPosBins/2, diam/3);

lengths  = linspace(0, maxLength, nLengths);



lineDictionary{kk, 1} = zeros(nPosBins, nTimeBins, nLines);
lineDictionary{kk, 2} = zeros(nLines, 2);
lineDictionary{kk ,3} = nTimeBins;

lineIdx = 1;
for ii = 1: numel(thetas)
    for jj = 1: numel(lengths)
        

        temp = zeros(nPosBins, nTimeBins);
        for t = 1:nTimeBins
            
            p = ceil((lengths(jj) - (t-t_mid) * cos(thetas(ii)))/sin(thetas(ii)) + p_mid);
            if p <= nPosBins && p >= 1
               temp(p, t) = 1;
            else
               temp(:, t) = nan;
            end
            
        end
        
        
        if all(isnan(sum(temp)))
            continue
        else
        
            lineDictionary{kk, 1}(:,:, lineIdx) = temp;

            lineDictionary{kk, 2}(lineIdx, :) = [thetas(ii) lengths(jj)];

            lineIdx = lineIdx + 1;
            
        end 
    end 
end


lineDictionary{kk, 1}(:,:, lineIdx+1:end) = [];
lineDictionary{kk, 2}(lineIdx+1:end, :)   = [];


end