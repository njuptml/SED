function [Sol, ind_zf] = DPC_nnLasso(X, y, Lambda, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential DPC rule proposed in
%  Two-Layer Feature Reduction for Sparse-Group Lasso via Decomposition of Convex Sets, arXiv,
%  Jie Wang and Jieping Ye. 
%
%% input: 
%         X: 
%			 the data matrix, each column corresponds to a feature 
%            each row corresponds to a data instance;
%
%         y: 
%			 the response vector
%
%         Lambda: 
%            the parameters sequence. 
%
%         opts: 
%            settings for the solver
%% output:
%         Sol: 
%              the solution. The ith column corresponds to the solution
%              with the ith parameter in Lambda
%
%         ind_zf: 
%              the index of the discarded features
%
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
p = size(X,2);
npar = length(Lambda); % number of parameters
opts.init = 0; % set .init for warm start

% --------------------- initialize the output --------------------------- %
Sol = zeros(p,npar);
ind_zf = zeros(p,npar);

% ------------------- compute the norm of each feature ------------------ %  
Xnorm = sqrt(sum(X.^2,1));

% --------------------------- compute lambda_max ------------------------ %
Xty = X'*y;
[lambda_max,indmx] = max(Xty);
opts.lambda_max = lambda_max;
xs = X(:,indmx);
if opts.rFlag == 1
    opts.rFlag = 0; % the parameter value passing to the solver is its 
                       % absolute value rather than a ratio
    Lambda = Lambda*lambda_max;
end

% ----------------- sort the parameters in descend order ---------------- %
[Lambdav,Lambda_ind] = sort(Lambda,'descend');
rLambdav = 1./Lambdav;

% ---------- solve the nnLasso problems sequentially with DPC ---------- %
lambdap = lambda_max;
for i = 1:npar    
    lambdac = Lambdav(1,i);
    if lambdac>=lambda_max
        Sol(:,Lambda_ind(i)) = 0; 
        ind_zf(:,Lambda_ind(i)) = 1;
        
    else
        rlambdac = rLambdav(1,i);
        if lambdap==lambda_max
            theta = y/lambda_max;
            n = xs;
        else
            theta = (y - X*Sol(:,Lambda_ind(i-1)))*rlambdap;
            n = y*rlambdap - theta;
        end
        
        n = n / norm(n);
        v = y*rlambdac - theta;
        Pv = v - n*((n'*v));
        o = theta + 0.5*Pv;
        phi = 0.5*norm(Pv);
		
		% ------- screening by DPC, remove the ith feature if T(i)=1 ----- %
        T = 1 - phi*Xnorm' > X'*o+1e-8;
        ind_zf(:,Lambda_ind(i)) = T;
        Xr = X(:,~T);
        
        if lambdap == lambda_max
            opts.x0 = zeros(size(Xr,2),1);
        else
            opts.x0 = Sol(~T,Lambda_ind(i-1));
        end
        
		% ----- solve the nnLasso problem on the reduced data matrix ----- %
        [x1, ~]= nnLeastR(Xr, y, lambdac, opts);
        
        Sol(~T,Lambda_ind(i)) = x1;     
        
        lambdap = lambdac;
        rlambdap = rlambdac;
    end
end

end

