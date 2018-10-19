function [Sol, ind_zf] = EDPP_Lasso(X, y, Lambda, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential EDPP rule proposed in
%  Jie Wang, Peter Wonka, and Jieping Ye,
%  Lasso Screening Rule via Dual Polytope Projection, 
%  Journal of Machine Learning Research.
%
%% input: 
%         X: 
%            the data matrix, each column corresponds to a feature 
%            each row corresponds to a data instance
%
%         y: 
%            the response vector
%       
%         Lambda: 
%            the parameters sequence
%
%         opts: 
%            settings for the solver
%% output:
%         Sol: 
%              the solution; the ith column corresponds to the solution
%              with the ith parameter in Lambda
%
%         ind_zf: 
%              the index of the discarded features; the ith column
%              refers to the solution of the ith value in Lambda
%
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
p = size(X,2);
npar = length(Lambda); % number of parameter values
opts.init = 0; % set .init for warm start

% --------------------- initialize the output --------------------------- %
Sol = zeros(p,npar);
ind_zf = zeros(p,npar);

% ------------------- compute the norm of each feature ------------------ %  
Xnorm = sqrt(sum(X.^2,1));

% --------------------------- compute lambda_max ------------------------ %
Xty = X'*y;
[lambda_max,indmx] = max(abs(Xty));
if opts.rFlag == 1
    opts.rFlag = 0; % the parameter value passing to the solver is its 
                       % absolute value rather than a ratio
    Lambda = Lambda*lambda_max;
end

% ----------------- sort the parameters in descend order ---------------- %
[Lambdav,Lambda_ind] = sort(Lambda,'descend');
rLambdav = 1./Lambdav;

% ----------- solve the Lasso problems sequentially with EDPP ----------- %
lambdap = lambda_max;
for i = 1:npar    
    lambdac = Lambdav(1,i);
    if lambdac>=lambda_max
        Sol(:,Lambda_ind(i)) = 0; 
        ind_zf(:,Lambda_ind(i)) = 1;      
    else
        if lambdap==lambda_max
            theta = y/lambdap;
            v = X(:,indmx);
            v1 = sign(v'*theta)*v;
        else
            theta = (y - X*Sol(:,Lambda_ind(i-1)))*rlambdap;
            v1 = y*rlambdap - theta;
        end
        
        rlambdac = rLambdav(1,i);
        
        v1 = v1 / norm(v1);
        v2 = y*rlambdac - theta;
        Pv2 = v2 - v1*((v1'*v2));
        o = theta + 0.5*Pv2;
        phi = 0.5*norm(Pv2);
		
		% ------- screening by EDPP, remove the ith feature if T(i)=1 ----- %
        T = 1 - phi*Xnorm' > abs(X'*o)+1e-8;
        ind_zf(:,Lambda_ind(i)) = T;
        Xr = X(:,~T);
        
        if lambdap == lambda_max
            opts.x0 = zeros(size(Xr,2),1);
        else
            opts.x0 = Sol(~T,Lambda_ind(i-1));
        end

		% ------ solve the Lasso problem on the reduced data matrix ------ %
        [x1, ~, ~]= LeastR(Xr, y, lambdac, opts);

        Sol(~T,Lambda_ind(i)) = x1;   
        
        lambdap = lambdac;
        rlambdap = rlambdac;
    end
end
end

