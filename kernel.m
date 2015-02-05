function [kerMat] = kernel(X, Y, kerType, sigma)

if(kerType == 1)
    kerMat = X * Y';
else if(kerType == 2)
%         x = rand(10,3);
%         y = rand(5,3);
        n1 = size(X, 1);
        n2 = size(Y, 1);

        sqTerm1 = repmat(sum(X.^2, 2), 1, n2);
        sqTerm2 = repmat(sum(Y.^2, 2)', n1, 1);
        D = (sqTerm1 + sqTerm2 - 2 * X * Y');
        kerMat = exp(-0.5 * D / sigma^2);
%         fullX = full(X);
%         fullY = full(Y);
%         newX = kron(ones(1,size(fullY,1)), fullX);
%         newY = reshape(fullY', 1, size(fullY,1) * size(fullY,2));
%         s2 = bsxfun(@minus, newX , newY).^2;
%         s2diff = reshape(reshape(sum(reshape(s2', size(fullY,2), size(fullY,1), []), 1), 1, size(fullX,1) * size(fullY,1) )', size(fullY,1), size(fullX,1))';
% %         sigma = 1;
%         kerMat = exp(-0.5 * s2diff / sigma^2);
    end
end

end

% function [kerMat] = kernel(spX, spY, kerType, sigma)
% 
% if(kerType == 1)
%     kerMat = spX * spY';
% else if(kerType == 2)
% %         x = rand(10,3);
% %         y = rand(5,3);
%         X = full(spX);
%         Y = full(spY);
%         newX = kron(ones(1,size(Y,1)), X);
%         newY = reshape(Y', 1, size(Y,1) * size(Y,2));
%         s2 = bsxfun(@minus, newX , newY).^2;
%         s2diff = reshape(reshape(sum(reshape(s2', size(Y,2), size(Y,1), []), 1), 1, size(X,1) * size(Y,1) )', size(Y,1), size(X,1))';
%         kerMat = exp(-0.5 * s2diff / sigma^2);
%     end
% end
% 
% end