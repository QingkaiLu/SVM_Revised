function [binaryPredT] = svmPredict(alpha, b, trainT, trainX, numAlpha0, predX, kerType, sigma)

%predT = alpha0' * diag(T0) * kernel(X0, predX, kerType) + alpha' * diag(trainT) * kernel(trainX, predX, kerType) + b;
%predT = predT';

numPredInsts = size(predX, 1);

if(numAlpha0 >= 1)
    allAlpha0 = cell(numAlpha0,1);
    allXAlpha0 = cell(numAlpha0,1);
    allTAlpha0 = cell(numAlpha0,1);
   for i = 1:numAlpha0
       alpha0Path = ['../alpha0/alpha0_', int2str(i), '.mat'];
       XAlpha0Path = ['../alpha0/X0_', int2str(i), '.mat'];
       tAlpha0Path = ['../alpha0/t0_', int2str(i), '.mat'];
       load(alpha0Path, 'alpha0');
       load(XAlpha0Path, 'XAlpha0');
       load(tAlpha0Path, 'TAlpha0');
       allAlpha0{i} = alpha0;
       allXAlpha0{i} = XAlpha0;
       allTAlpha0{i} = TAlpha0;
   end
end

if(numAlpha0 >= 1)
    avgVec = zeros(1, numPredInsts);
    for i = 1:numAlpha0
        alpha0 = allAlpha0{i};
        XAlpha0 = allXAlpha0{i};
        TAlpha0 = allTAlpha0{i};
%         tmp = alpha0' * diag(TAlpha0) * kernel(XAlpha0, predX, kerType);
%         size(tmp)
%         size(XAlpha0)
%         size(predX)
        avgVec = avgVec + alpha0' * diag(TAlpha0) * kernel(XAlpha0, predX, kerType, sigma);
    end
    avgVec = avgVec / numAlpha0;
    predT = avgVec + alpha' * diag(trainT) * kernel(trainX, predX, kerType, sigma) + b;

else
    predT = alpha' * diag(trainT) * kernel(trainX, predX, kerType, sigma) + b;
end

predT = predT';
binaryPredT = (predT > 0) + -1 * (predT <= 0);


end

