clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example for integrating EDPP screening rule with SLEP to
% solve the Lasso problem at a given sequence of parameter values:
%
%  min  1/2 || X * beta - y||^2 + lambda * ||beta||_1
%
% Author: Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate the data                     
m=250;  n=10000;     % The data matrix is of size m x n
X = normrnd(0,1,[m,n]);
y = normrnd(0,1,[m,1]);

%% set up the solver from SLEP, can leave as default
opts=[];

% termination criterion
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=1000;   % maximum number of iterations
					 % when the improvement is small, 
					 % SLEP may stop without running opts.maxIter steps

% normalization
opts.nFlag=0;       % without normalization

% regularization
opts.rFlag=1;       % the input parameter 'lambda' is a ratio in (0, 1]

opts.fName = 'LeastR';  % compute a sequence of lasso problems

opts.mFlag=0;       % treating it as compositive function 
opts.lFlag=0;       % Nemirovski's line search

%% set the regularization paramter values
% if the parameter values are the ratios of lambda/lambda_max; if use the 
% absolute value, please set opts.rFlag = 0
%
ub = 1; % upper bound of the parameter values
lb = 0.05; % lower bound of the parameter values
npar = 100; % number of parameter values
delta_lambda = (ub - lb)/(npar-1);
lambda=lb:delta_lambda:ub; % the parameter sequence

%% solve the Lasso problems along the sequence of parameter values

tic
[Sol, ind_zf] = EDPP_Lasso(X, y, lambda, opts);
toc


