function [ ] = plotResults()
clear; clc;
% t = -pi : .1 : pi;
% x = sin(t);
% y = cos(t);
% plot(t,x,'color','r'); hold on;
% plot(t,y,'color','b');
% legend('sin','Cos', 'Location','NW');

 dataName = 'w1a';
%dataName = 'mushrooms';
% dataName = 'a7a';

% Compute the accuracy from delta matrix
load(['results/' dataName 'TestData'],'D_test');
load(['results/' dataName '_deltamatrix'],'Delta');
load(['results/' dataName '_BV'], 'BV');

accCell = cell(size(Delta));
accEnsmbVec = zeros(size(Delta));
biasEnsmbMat = zeros(size(Delta));
uVarEnsmbMat = zeros(size(Delta));
bVarEnsmbMat = zeros(size(Delta));
nVarEnsmbMat = zeros(size(Delta));

for i = 1:size(Delta, 1)
    for j = 1:size(Delta, 2)
        predT = Delta{i,j};
        testT = D_test(:,1);
        acc = zeros(1, size(predT, 2));
        
        for k = 1:size(predT, 2)
            acc(k) = sum(predT(:,k)  == testT) / size(testT, 1);
        end
        accCell{i,j} = acc;
        accEnsmbVec(i, j) = mean(acc);
        
        bv = BV{i,j};
        biasEnsmbMat(i, j) = bv(1);
        uVarEnsmbMat(i, j) = bv(2);
        bVarEnsmbMat(i, j) = bv(3);
        nVarEnsmbMat(i, j) = bv(4);
        
    end
end

save(['results/', dataName 'AccCell'],'accCell');
% accEnsmbVec

% C_vector = [0.001,0.01,0.1,1,10,100,1000];
C_vector = -3:3;
numAlpha0_vector = [0,1,2,4,8,16,32];

for i = 1:size(numAlpha0_vector, 2)
    figure;
%     1 - accEnsmbVec(:, i)
    
    plot(C_vector, 1 - accEnsmbVec(:, i),'color','r','LineWidth',2); hold on;
    plot(C_vector,biasEnsmbMat(:, i),'color','g','LineWidth',2);
    plot(C_vector,uVarEnsmbMat(:, i),'color','b','LineWidth',2);
    plot(C_vector,bVarEnsmbMat(:, i),'color','m','LineWidth',2);
    plot(C_vector,nVarEnsmbMat(:, i),'color','c','LineWidth',2);
    legend('avg. error', 'bias', 'unbiased variance', 'biased variance', 'net variance', 'Location','NW');
    xlabel('log C');
%     title(['number of runs of common svm to generate w0 is: ' int2str(numAlpha0_vector(i))]);
    if(i > 1)
        title([int2str(numAlpha0_vector(i)) ' runs of common svm to generate w0']);
    else
        title('Common svm');
    end
    hold off;
    
    Image = getframe(gcf);
    figurePath = ['plots/' dataName 'Alpha0_' int2str(numAlpha0_vector(i)) '.png'];
    imwrite(Image.cdata, figurePath);
    hndl = gcf();
    close(hndl);
    hold off;
end

for i = 2:size(numAlpha0_vector, 2)
    figure;
%     1 - accEnsmbVec(:, i)
    plot(C_vector,accEnsmbVec(:, 1) - accEnsmbVec(:, i),'color','r','LineWidth',2); hold on;
    plot(C_vector,biasEnsmbMat(:, i) - biasEnsmbMat(:, 1),'color','g','LineWidth',2);
    plot(C_vector,uVarEnsmbMat(:, i) - uVarEnsmbMat(:, 1),'color','b','LineWidth',2);
    plot(C_vector,bVarEnsmbMat(:, i) - bVarEnsmbMat(:, 1),'color','m','LineWidth',2);
    plot(C_vector,nVarEnsmbMat(:, i) - nVarEnsmbMat(:, 1),'color','c','LineWidth',2);
    legend('avg. error change', 'bias change', 'unbiased variance change', 'biased variance change', 'net variance change', ...
        'Location','SW');
    xlabel('log C');
    if(i > 1)
        title([int2str(numAlpha0_vector(i)) ' runs of common svm to generate w0']);
    else
        title('Common svm');
    end
    hold off;
    
    Image = getframe(gcf);
    figurePath = ['plots/' dataName 'ChangeAlpha0_' int2str(numAlpha0_vector(i)) '.png'];
    imwrite(Image.cdata, figurePath);
    hndl = gcf();
    close(hndl);
    hold off;
end



end

