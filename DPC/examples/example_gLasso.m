 clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example for integrating EDPP screening rule with SLEP to
% solve the group Lasso problem at a given sequence of parameter values:
%
%  min  1/2 || X * beta - y||^2 + lambda * sum_i w_i * ||beta_i||_2
%
% Related papers
%
% [1] Jie Wang, Peter Wonka, and Jieping Ye, Lasso Screening Rules via Dual
%     Polytope Projection, Journal of Machine Learning Research, to appear
%
% Author: Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% generate the data                     
m=250;  n=10000;     % The data matrix is of size m x n
X = normrnd(0,1,[m,n]);
y = normrnd(0,1,[m,1]);

% ----------------------- set the group information --------------------- %
ind = 0:10:n; % the group index; the ith group include ind(i)+1:ind(i+1) features
sg = diff(ind); % the size of each group
w = sqrt(sg)'; % the weight of each group

%----------------------- Set optional items -----------------------
opts=[];

% solver for group Lasso
opts.fName = 'glLeastR';  

% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=1000;   % maximum number of iterations
					 % when the improvement is small, 
					 % SLEP may stop without running opts.maxIter steps

% Normalization
opts.nFlag=0;       % without normalization

% Regularization
opts.rFlag=1;       % the input parameter 'lambda' is a ratio in (0, 1]

% Group Property
opts.ind=ind;       % set the group indices
opts.q=2;           % set the value for q
opts.sWeight=[1,1]; % set the weight for positive and negative samples
opts.gWeight=w;
                    % set the weight for the group, a cloumn vector
                    
opts.mFlag=0;       % treating it as compositive function 
opts.lFlag=0;       % Nemirovski's line search

%% set the regularization parameter values
% if the parameter values are the ratios of lambda/lambda_max; if use the 
% absolute value, please set opts.rFlag = 0
%
ub = 1; % upper bound of the parameter values
lb = 0.05; % lower bound of the parameter values
npar = 100; % number of parameter values
delta_lambda = (ub - lb)/(npar-1);
lambda=lb:delta_lambda:ub; % the parameter sequence

%% solve the group Lasso problems along the sequence of parameter values

tic
[Sol, ind_zg] = EDPP_gLasso(X, y, lambda, opts);
toc



    









