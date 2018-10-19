function [LambdaLa_mx, indLambdaLa_mx] = getLambdaLa_mx(X,y,Alpha,ind)
%% compute lambda_mx for given alpha

nAlpha = length(Alpha);
LambdaLa_mx = zeros(1,nAlpha);
indLambdaLa_mx = zeros(1,nAlpha);
ng = size(ind,2); % the number of groups
sg = ind(2,:)-ind(1,:)+1; % the size of groups
Xty = X'*y;
aXty = abs(Xty);
rho = zeros(ng,1);

for i = 1 : nAlpha
    alpha = Alpha(i);    
    for g = 1:ng
        gamma = zeros(sg(g),1);
        z = sort(aXty(ind(1,g):ind(2,g)),'descend'); % sort Xgty in absolute value
        nzz = nnz(z); % the number of nonzero elements in z
        Xgty = Xty(ind(1,g):ind(2,g));
        Xgtyz = repmat(Xgty,1,nzz)./repmat((z(1:nzz))',sg(g),1);
        sXgtyz = shrinkage(Xgtyz,1);
        gamma(1:nzz) = (sqrt(sum(sXgtyz.^2,1)))';
        if nzz~=sg(g)
            gamma(nzz+1:end)=inf;
        end
        
        c = alpha*ind(3,g);
        [Lia,Locb] = ismember(c,gamma);
        if Lia == 1
            rho(g) = gamma(Locb);
        else
            k = find(gamma<c,1,'last');
            zk = z(1:k);
            r = roots([k-c*c, -2*norm(zk,1), zk'*zk]);
            if k<sg(g)
                rho(g)=r(r<z(k)&r>z(k+1));
            else
                rho(g)=r(r<z(k)&r>0);
            end
        end
    end
    [LambdaLa_mx(i), indLambdaLa_mx(i)] = max(rho);
end

end

