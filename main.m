function [ ] = main( )
clc
clear
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
D_name = 'w1a';
% [trainT, trainX] = libsvmread('../data/a1a');
% [testT, testX] = libsvmread('../data/a1a.t');
% testX = testX(:, 1:size(trainX, 2));

[trainT, trainX] = libsvmread(['../data/' D_name]);
[testT,testX] = libsvmread('../data/w1a.t');
Data1 = [trainT trainX];
Data2 = [testT testX];
Data2 = Data2(:,1:size(Data1,2));
Data = vertcat(Data1,Data2);
split_size = 2000;

% [trainT, trainX] = libsvmread(['../data/' D_name]);
% label_idx = trainT == 2; 
% trainT(label_idx) = -1;
% Data = [trainT trainX];
% split_size = 2000;


split_time = 10;
AC = cell(1,split_time);
for s = 1:1:split_time % random split 5 times
    fprintf('%dth random split and calculate accuracy\n',s);
    tic
    [D_train, idx] = datasample(Data, split_size, 'Replace', false);
    D_test = Data;
    D_test(idx,:) = [];
    testT = D_test(:,1);
    testX = D_test(:,2:end);
    trainX = D_train(:,2:end);
    trainT = D_train(:,1);

    % testT = trainT(split_size:end,:);
    % testX = trainX(split_size:end,:);
    % trainX = trainX(1:split_size,:);
    % trainT = trainT(1:split_size,:);

    % Calculate the accuracy of SVM(numAlpha0 = 0) and revised SVM(numAlpha0 ~= 0)
    % tic
    numAlpha0_vector = [0,1,2,4,8,16,32];
    C_vector = [0.001,0.01,0.1,1,10,100,1000];
    kerType = 1;
    accuracy = zeros(length(C_vector),length(numAlpha0_vector));

    for i = 1:1:length(C_vector)
        rmdir('../alpha0','s');
        
        mkdir('../alpha0');
        C = C_vector(i);
        genAlpha0(trainT, trainX, max(numAlpha0_vector), C, kerType);
        for j = 1:1:length(numAlpha0_vector) 
            numAlpha0 = numAlpha0_vector(j);
            [alpha, b] = svmTrain(trainT, trainX, numAlpha0, C, kerType);
            predT = svmPredict(alpha, b, trainT, trainX, numAlpha0, testX, kerType);
            accuracy(i,j) = sum(predT == testT) / size(testT, 1);
        end
    end
    % toc
    % accuracy
    AC{1,s}=accuracy;
    toc
    fprintf('%dth random split and calculation finished!\n',s);
    save([D_name '_accuracy'],'AC');
end
% calculate average accuracy
ac = AC{1,1};
for i = 2:1:split_time
    ac = plus(ac,AC{1,i});
end
ac = ac./split_time;
save([D_name '_accuracy_average'], 'ac');

disp('accuracy hold it here')



end

