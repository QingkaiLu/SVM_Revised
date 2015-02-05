function [ ] = BV_decomposition_Gaus(numAlpha0)
% clc
% clear
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

trainDataName = 'w1a';
D_name = [trainDataName, 'GausAlpha', int2str(numAlpha0)];

% [trainT, trainX] = libsvmread(['../data/' D_name]);
% % change label '2' into -1
% label_idx = trainT == 2; 
% trainT(label_idx) = -1;
% 
% split_size = 4000;
% Data = [trainT trainX];
% [D_train, idx] = datasample(Data, split_size, 'Replace', false);
% D_test = Data;
% D_test(idx,:) = [];
% testT = D_test(:,1);
% testX = D_test(:,2:end);
% trainX = D_train(:,2:end);
% trainT = D_train(:,1);

[trainT, trainX] = libsvmread(['../data/' trainDataName]);
[testT, testX] = libsvmread('../data/w1a.t');
Data1 = [trainT trainX];
Data2 = [testT testX];
Data2 = Data2(:,1:size(Data1,2));
Data = vertcat(Data1,Data2);
% split_size = 25000;
split_size = ceil(0.5 * size(Data,1));
[D_train, idx] = datasample(Data, split_size, 'Replace', false);
D_test = Data;
D_test(idx,:) = [];
testT = D_test(:,1);
testX = D_test(:,2:end);
save([D_name 'testData'],'D_test');


n = 50;  % n is number of classifiers
m = 1000;  % m is the number of points trained for each classifer

% C_vector = [0.001,0.01,0.1,1,10,100,1000];
C_vector = [0.01,0.1,1,10,100];
sigma_vector = [0.01,0.1,1,10,100];
% numAlpha0 = 16;
% C_vector = [0.001,0.01,0.1,1];
% numAlpha0_vector = [0,1,2,4,8,16,32];

% here seprate D_train in to 'n' learning sets, which has m data points.
for i = 1:1:n
    D_n{i} = datasample(D_train, m, 'Replace',true);
end

% Calculate Bias, variance and error with chaging C
kerType = 2;
% Delta matrix stores predT for every C and alpha0
Delta = cell(length(C_vector),length(sigma_vector));
accCell = cell(length(C_vector),length(sigma_vector));
% try 
%     load([D_name '_deltamatrix.mat']); 
% catch
    for i = 1:1:length(C_vector)
        for j = 1:1:length(sigma_vector)
            tic
            C = C_vector(i);
            sigma = sigma_vector(j);
            fprintf('C=%f and alpha0=%d delta matrix calculation begin!\n',C,numAlpha0);
            predT = zeros(length(testT),n);
            acc = zeros(1,n);
            for k = 1:1:n
                rmdir('../alpha0','s');
                mkdir('../alpha0');
                trainX = D_n{k}(:,2:end);
                trainT = D_n{k}(:,1);
                if(numAlpha0 > 0)
                    genAlpha0(trainT, trainX, numAlpha0, C, kerType, sigma);
                end
                [alpha, b] = svmTrain(trainT, trainX, numAlpha0, C, kerType, sigma);
                predT(:,k) = svmPredict(alpha, b, trainT, trainX, numAlpha0, testX, kerType, sigma);
                acc(k) = sum(predT(:,k)  == testT) / size(testT, 1);
            end
            Delta{i,j} = predT;
            save([D_name '_deltamatrix'],'Delta');
            fprintf('C=%f and alpha0=%d delta matrix calculation finished!\n',C,numAlpha0);
            accCell{i,j} = acc;
            save([D_name 'accCell'],'accCell');
            toc
        end
    end
%     save([D_name '_deltamatrix'],'Delta');
% end
% BV is the cell array store average BIAS, Vu, Vb, Vn for each (C,numAlpha)
% each cell 'bv' is a vector of [Bias, Vu, Vb, Vn]
test_length = length(testT);
BV = cell(length(C_vector),length(sigma_vector));       
weightX = ones(test_length,1);
for i = 1:1:length(C_vector)
    for j = 1:1:length(sigma_vector)
        predT = Delta{i,j};
        Bias = zeros(test_length,1);
        Variance_u = zeros(test_length,1);
        Variance_b = zeros(test_length,1);
%         Variance_n = zeros(length(testT),1);
        for k = 1:1:test_length % loop over all the data points in test
            fprintf('debug k=%d\n',k);
            pred_row = predT(k,:);
            % calculate the main prediction of one data point
            ym = sum(pred_row);
            if ym >= 0
                ym = 1;
            else
                ym = -1;
            end
            Bias(k,1) = abs( (ym - testT(k,1))/2 );
            if ym == testT(k,1) % unbiased (ym = t)
                ind = find(pred_row ~= ym); % ind is the index that ym ~= t
                Variance_u(k,1) = length(ind)/n;
                Variance_b(k,1) = 0;
            else  % biased (ym ~= t)
                ind = find(pred_row ~= ym); % ind is the index that ym ~= t
                Variance_b(k,1) = length(ind)/n;
                Variance_u(k,1) = 0;
            end
        end
        bv = [weightX'*Bias/test_length, weightX'*Variance_u/test_length, weightX'*Variance_b/test_length];
        bv(1,4) = bv(1,2) - bv(1,3);
        BV{i,j} = bv;
    end
end
save([D_name '_BV'],'BV');

BV


end