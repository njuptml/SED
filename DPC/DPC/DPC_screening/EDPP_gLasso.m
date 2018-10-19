function [Sol, ind_zgd] = EDPP_gLasso(X, y, Lambda, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential EDPP rule proposed in
%  Lasso Screening Rule via Dual Polytope Projection, JMLR
%  Jie Wang, Peter Wonka, and Jieping Ye. 
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
%         ind_zgd: 
%              the index of the discarded groups; the ith column
%              refers to the solution of the ith value in Lambda
%
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = size(X,2);
npar = length(Lambda); % number of parameters

% ---------------------------- pass parameters -------------------------- %
opts.init = 0; % set .init for warm start
gind = opts.ind;
sg = diff(gind); % size of each group
gWeight = opts.gWeight;
ng = length(gWeight); % number of groups

% --------------------- initialize the output --------------------------- %
Sol = zeros(p,npar);
ind_zgd = zeros(ng,npar);

% ------- construct sparse matrix to vectorize the computation ---------- %
fg_ind = zeros(1,p);
for i = 1:ng
    fg_ind(1,gind(i)+1:gind(i+1))=i; % fg_ind(j) is the index of the group 
                                     % which contains the jth feature
end
gS = sparse(fg_ind,1:p,ones(1,p),ng,p,p); % ith row refers to the ith group, if the ith row
                   % refers to the i_s to i_e features, then
                   % gS(i,i_s:i_e) = 1.

% ------ compute the norm of each submatrix Xg and ||Xg'*y||_2 ---------- %  
Xgnorm = zeros(1,ng);
for i = 1 : ng
    Xgnorm(i) = norm(X(:,gind(i)+1:gind(i+1)),2);
end
       
% --------------------------- compute lambda_max ------------------------ %
Xty = X'*y;
Xgtynorm = sqrt(gS*(Xty.*Xty));
[lambda_max, indmx] = max(Xgtynorm./gWeight);
opts.lambda_max = lambda_max;
Xs = X(:,gind(indmx)+1:gind(indmx+1));

if opts.rFlag == 1
    opts.rFlag = 0; % the parameter value passing to the solver is its 
                       % absolute value rather than a ratio
    Lambda = Lambda*lambda_max;
end

% ----------------- sort the parameters in descend order ---------------- %
[Lambdav,Lambda_ind] = sort(Lambda,'descend');
rLambdav = 1./Lambdav;

% ----------------- solve group Lasso sequentially with EDPP ------------ %
lambdap = lambda_max;
for i = 1:npar
    lambdac = Lambdav(1,i);
    if lambdac>=lambda_max
        Sol(:,Lambda_ind(i)) = 0; 
        ind_zgd(:,Lambda_ind(i)) = 1;        
    else
        if lambdap==lambda_max
            theta = y/lambda_max;
            v1 = Xs*(Xs'*y);
        else
            theta = (y - X*Sol(:,Lambda_ind(i-1)))*rlambdap;
            v1 = y*rlambdap-theta;
        end
        
        rlambdac = rLambdav(1,i);
        
        v1 = v1/norm(v1);
        v2 = y*rlambdac-theta;
        Pv2 = v2 - (v1'*v2)*v1;
        o = theta + 0.5*Pv2;
        Xto = X'*o;
        r = 0.5*norm(Pv2);

        % ------- screening by EDPP, remove the ith group if T(i)=1 ----- %
        T = gWeight-r*Xgnorm' > sqrt(gS*(Xto.*Xto)) + 1e-8;      
        ind_zgd(:,Lambda_ind(i)) = T;
        ind_rf = sum(gS(T==0,:),1)==1; % index of remaining features
        Xr = X(:,ind_rf);
        
        sgr = sg(T==0);
        ngr = length(sgr);
        indr = zeros(1,ngr+1);
        indr(1,2:end)= cumsum(sgr);
        opts.ind = indr;
        opts.gWeight = gWeight(T==0);
        if lambdap == lambda_max
            opts.x0 = zeros(size(Xr,2),1);
        else
            opts.x0 = Sol(ind_rf,Lambda_ind(i-1));
        end
        
		% --- solve the group Lasso problem on the reduced data matrix -- %
        [x1, ~, ~]= glLeastR(Xr, y, lambdac, opts);   
        
        Sol(ind_rf,Lambda_ind(i)) = x1;
        
        lambdap = lambdac;
        rlambdap = rlambdac;
    end
end
end

