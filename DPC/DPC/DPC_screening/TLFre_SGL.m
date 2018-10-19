function [Sol, ind_zf1, ind_zf2] = TLFre_SGL(X, y, Alpha, LambdaL, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implementation of the sequential TLFre rule proposed in
%  Jie Wang and Jieping Ye,
%  Two-Layer Feature Reduction for Sparse-Group
%  Lasso via Decomposition of Convex Sets,
%  NIPS 2014.
%
%% input: 
%         X: 
%            the data matrix, each column corresponds to a feature 
%            each row corresponds to a data instance
%
%         y: 
%            the response vector
%
%         Alpha:
%            the parameter values of alpha
%       
%         LambdaL: 
%            the parameter values of lambda
%
%         opts: 
%            settings for the solver
%% output:
%         Sol: 
%              the solution that is a 3D tensor; Sol(:,i,k) stores the the 
%              solution with the kth values in Alpha and ith values in LambdaL  
%
%         ind_zf1: 
%              a 3D tensor that stores the index of the discarded features 
%              by the first layer of TLFre; ind_zf1(:,i,k) the index of 
%              discarded features corresponding to the kth values in Alpha 
%              and ith values in LambdaL  
%
%         ind_zf2: 
%              a 3D tensor that stores the index of the discarded features 
%              by the second layer of TLFre; ind_zf2(:,i,k) the index of 
%              discarded features corresponding to the kth values in Alpha 
%              and ith values in LambdaL  
%
%% For any problem, please contact Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
p = size(X,2);
nAlpha = length(Alpha); % number of parameter values of alpha
nLambdaL = length(LambdaL); % number of parameter values of lambda
opts.init=2;        % starting from a zero point

% --------------------- initialize the output --------------------------- %
Sol = zeros(p,nLambdaL,nAlpha);
ind_zf1 = zeros(p,nLambdaL,nAlpha);
ind_zf2 = zeros(p,nLambdaL,nAlpha);

% ---------- compute the effective region of alpha and lambdaL ---------
[LambdaLa_mx, indLambdaLa_mx] = getLambdaLa_mx(X,y,Alpha,opts.ind);

% --------- compute the norm of each feature and each submatrix
Xnorm = sqrt(sum(X.^2,1));
ind = opts.ind;
ng = size(ind,2);
Xgnorm = zeros(1,ng);
for i = 1 : ng
    Xgnorm(i) = norm(X(:,ind(1,i):ind(2,i)));
end

% ------- construct sparse matrix to vectorize the computation ----------
gind = opts.gind;
G = sparse(gind,1:p,ones(1,p),ng,p);

% ----------- solve SGL sequentially via TLFre ------------------
Xty = X'*y;
opts.rFlag = 0; % the input parameters are their true values
T2 = false(p,1);
supg = zeros(ng,1); % the supreme values of the 1st layer 
supf = zeros(p,1); % the supreme values of the 2st layer 
for i = 1 : nAlpha
    alpha = Alpha(i);
    lambdaL_mx = LambdaLa_mx(i);
    lambdaLp = lambdaL_mx;
    tLambdaL = LambdaL*lambdaL_mx; % the true value of LambdaL
    [dtLambdaL, indd] = sort(tLambdaL, 'descend');
    for j = 1 : nLambdaL
        lambdaLc = dtLambdaL(j);
        % --- estimate the possible region for the dual optimal solution --
        % -- use beta^*(lambdaL^j,alpha) to estimate beta^*(lambdaL^(j+1),alpha)
        if lambdaLc < lambdaL_mx % if the (lambdaGL,lambdaLc) are in the effective region
            if lambdaLp == lambdaL_mx
                theta = y/lambdaL_mx;
                indmx = indLambdaLa_mx(i);
                Xmx = X(:,ind(1,indmx):ind(2,indmx)); 
                n = Xmx*shrinkage(Xty(ind(1,indmx):ind(2,indmx))/lambdaL_mx, 1);
            else
                theta = (y-X*Sol(:,indd(j-1),i))/lambdaLp;
                n = y/lambdaLp-theta;
            end
            
            n = n/norm(n);
            v = y/lambdaLc-theta;
            Pv = v - (v'*n)*n;
            o = theta+0.5*Pv;
            r = 0.5*norm(Pv);
            
            % --------- the first layer screening -------------
            c = X'*o; % Xg'*o is the center for gth group
            csm = sparse(1:p,gind,c,p,ng); % split c to construct a sparse matrix, cg is in the gth column
            r1  = r * Xgnorm;
            cmx = max(abs(csm)); % compute |cg|_{infty}
            TT = cmx<1;
            supg(TT) = subplus(cmx(TT)+r1(TT)-1);
            TT = ~TT;
            if nnz(TT)>1
                sc = shrinkage(sum(csm(:,TT),2),1);
                supg(TT)=sqrt(G(TT,:)*(sc.*sc))+r1(TT)';
            elseif nnz(TT)==1
                sc = shrinkage(csm(:,TT),1);
                supg(TT)=sqrt(G(TT,:)*(sc.*sc))+r1(TT)';
            end
            
            Tg = supg<alpha*ind(3,:)';
            if nnz(Tg)>1
                T1 = (logical(sum(G(Tg,:))))';
            elseif nnz(Tg)==1
                T1 = (logical(G(Tg,:)))';
            else
                T1 = false(p,1);
            end
            ind_zf1(:,indd(j),i) = T1;
            
            % --------- the second layer screening ------------
            supf(~T1) = abs(c(~T1))+(Xnorm(~T1))'*r;
            T2(~T1) = supf(~T1)<=1;
            T2(T1) = false;
            ind_zf2(:,indd(j),i) = T2;
            
            T = T1|T2;
            
            % -------- construct the reduced data matrix -------
            rX = X(:,~T);
            
            % -------- get the reduced group information -------
            rgind = gind(~T);
            [ugind,istart,~] = unique(rgind);
            rind = zeros(3,length(ugind));
            rind(1,:) = istart';
            rind(2,:) = [(istart(2:end)-1)', length(rgind)];
            rind(3,:) = ind(3,ugind);
            optsloc = opts;
            optsloc.ind = rind;
            
            % -------- solve the reduced SGL problem ---------
            z=[lambdaLc,alpha*lambdaLc];
            [x1, ~, ~]= sgLeastR(rX, y, z, optsloc);
           
            
            % ------- recover the solution -----------
            Sol(~T,indd(j),i) = x1;    
            
            lambdaLp = lambdaLc;
        else
            ind_zf1(:,indd(j),i) = true(p,1);
        end                
    end
end
end

