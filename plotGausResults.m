function [ ] = plotGausResults()
clear; clc;

 dataName = 'w1aGausAlpha8';
%dataName = 'mushrooms';
% dataName = 'a7a';

% Compute the accuracy from delta matrix
load(['results/' dataName 'testData'],'D_test');
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
% for i = 2:size(numSigma_vector, 2)
%     figure;
% %     1 - accEnsmbVec(:, i)
%     plot(C_vector, 1 - accEnsmbVec(:, i),'color','r','LineWidth',2); hold on;
%     plot(C_vector,biasEnsmbMat(:, i),'color','g','LineWidth',2);
%     plot(C_vector,uVarEnsmbMat(:, i),'color','b','LineWidth',2);
%     plot(C_vector,bVarEnsmbMat(:, i),'color','m','LineWidth',2);
%     plot(C_vector,nVarEnsmbMat(:, i),'color','c','LineWidth',2);
%     plot(C_vector,biasEnsmbMat(:, i) - biasEnsmbMat(:, 1),'color','k','LineWidth',2);
%     plot(C_vector,nVarEnsmbMat(:, i) - nVarEnsmbMat(:, 1),'color','y','LineWidth',2);
%     legend('avg. error', 'bais', 'unbiased variance', 'biased variance', 'net variance', ...
%     'bias change', 'nvar change', 'Location','NW');
%     xlabel('Log C');
%     hold off;
%     
%     Image = getframe(gcf);
%     figurePath = ['plotsGaus/' dataName 'BiasAlpha0_' int2str(numSigma_vector(i)) '.png']
%     imwrite(Image.cdata, figurePath);
%     hndl = gcf();
%     close(hndl);
%     hold off;
% end

% C_vector = [0.001,0.01,0.1,1,10,100,1000];
C_vector = -2:2;
% numAlpha0_vector = [0,1,2,4,8,16,32];
numSigma_vector = -2:2;
% numSigma_vector = [0.01,0.1,1,10,100];

for i = 1:size(numSigma_vector, 2)
    figure;
%     1 - accEnsmbVec(:, i)
    plot(C_vector, 1 - accEnsmbVec(:, i),'color','r','LineWidth',2); hold on;
    plot(C_vector,biasEnsmbMat(:, i),'color','g','LineWidth',2);
    plot(C_vector,uVarEnsmbMat(:, i),'color','b','LineWidth',2);
    plot(C_vector,bVarEnsmbMat(:, i),'color','m','LineWidth',2);
    plot(C_vector,nVarEnsmbMat(:, i),'color','c','LineWidth',2);
    legend('avg. error', 'bias', 'unbiased variance', 'biased variance', 'net variance', 'Location','Best');
    xlabel('Log C');
%     title(['Common SVM with log10(sigma)=' int2str(numSigma_vector(i)) ]);
    title(['Our changed SVM with log10(sigma)=' int2str(numSigma_vector(i)) ]);
    hold off;
    
    Image = getframe(gcf);
    figurePath = ['plotsGaus/' dataName 'Sigma_' int2str(numSigma_vector(i)) '.png']
    imwrite(Image.cdata, figurePath);
    hndl = gcf();
    close(hndl);
    hold off;
end

% for i = 2:size(numSigma_vector, 2)
%     figure;
% %     1 - accEnsmbVec(:, i)
%     plot(C_vector, 1 - accEnsmbVec(:, i),'color','r','LineWidth',2); hold on;
%     plot(C_vector,biasEnsmbMat(:, i),'color','g','LineWidth',2);
%     plot(C_vector,uVarEnsmbMat(:, i),'color','b','LineWidth',2);
%     plot(C_vector,bVarEnsmbMat(:, i),'color','m','LineWidth',2);
%     plot(C_vector,nVarEnsmbMat(:, i),'color','c','LineWidth',2);
%     plot(C_vector,biasEnsmbMat(:, i) - biasEnsmbMat(:, 1),'color','k','LineWidth',2);
%     plot(C_vector,nVarEnsmbMat(:, i) - nVarEnsmbMat(:, 1),'color','y','LineWidth',2);
%     legend('avg. error', 'bais', 'unbiased variance', 'biased variance', 'net variance', ...
%     'bias change', 'nvar change', 'Location','NW');
%     xlabel('Log C');
%     hold off;
%     
%     Image = getframe(gcf);
%     figurePath = ['plotsGaus/' dataName 'BiasAlpha0_' int2str(numSigma_vector(i)) '.png']
%     imwrite(Image.cdata, figurePath);
%     hndl = gcf();
%     close(hndl);
%     hold off;
% end



end

