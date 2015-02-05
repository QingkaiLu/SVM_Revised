%reference: http://cmp.felk.cvut.cz/cmp/software/stprtool/manual/svm/list/cc.html

function [alpha, b] = svmTrain(T, X, numAlpha0, C, kerType, sigma)

%[label_vector, instance_matrix] = libsvmread('../data/a1a')

H = diag(T) * kernel(X, X, kerType, sigma) * diag(T);
%H = H + 1e-12*eye(size(H));
% tmpH = kernel(X, X, kerType).*(T*T');
% eq = isequal(H, tmpH)
numInsts = size(T, 1);

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
    %    f = -ones(1, size(t, 1)) + diag(t) * kernel(X, X0) * diag(t0) * alpha0;
    avgVec = zeros(numInsts, 1);

    for i = 1:numAlpha0
        alpha0 = allAlpha0{i};
        XAlpha0 = allXAlpha0{i};
        TAlpha0 = allTAlpha0{i};
        avgVec = avgVec + kernel(X, XAlpha0, kerType, sigma) * diag(TAlpha0) * alpha0;
    end

    avgVec = avgVec / numAlpha0;
    f = -ones(numInsts, 1) + diag(T) * avgVec;

else
    f = -ones(numInsts, 1);
end

A = [];
b2 = [];
% Aeq = diag(T);
% beq = zeros(numInsts, 1);
Aeq = T';
beq = 0;
lb = zeros(numInsts, 1);
ub = C * ones(numInsts, 1);
%x0 = zeros(numInsts, 1);

%qp_options = optimset('Display','off');
qp_options = optimset('Algorithm', 'interior-point-convex','Display','off');
%alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0, qp_options);
alpha = quadprog(H,f,A,b2,Aeq,beq,lb,ub,[], qp_options);

marginInstsIdx = (alpha > 0 & alpha < C);
marginT = T(marginInstsIdx);
marginX = X(marginInstsIdx, :);
% b = 1 / size(marginInstsIdx, 1) * sum(marginT - (alpha0' * diag(t0) * kernel(X0, marginX))' - (alpha' * diag(t) * kernel(X, marginX))');

b = sum(marginT - (alpha' * diag(T) * kernel(X, marginX, kerType, sigma))') / size(marginInstsIdx, 1) ;

end

