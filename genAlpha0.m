function [ ] = genAlpha0(T, X, numAlpha0, C, kerType, sigma)
    
numInsts = size(T, 1);
samplePercent = 0.5;
for i = 1:numAlpha0
   alpha0Path = ['../alpha0/alpha0_', int2str(i), '.mat'];
   XAlpha0Path = ['../alpha0/X0_', int2str(i), '.mat'];
   tAlpha0Path = ['../alpha0/t0_', int2str(i), '.mat'];
   %generate each XAlpha0 and TAlpha0
%    sampleIdx = datasample(1:numInsts, uint8(numInsts * samplePercent), 'Replace', false);
   sampleIdx = datasample(1:numInsts, floor(numInsts * samplePercent), 'Replace', false);
   XAlpha0 = X(sampleIdx, :);
   TAlpha0 = T(sampleIdx);
   [alpha0, ~] = svmTrain(TAlpha0, XAlpha0, 0, C, kerType, sigma);
   save(alpha0Path, 'alpha0');
   save(XAlpha0Path, 'XAlpha0');
   save(tAlpha0Path, 'TAlpha0');
end
   
end

