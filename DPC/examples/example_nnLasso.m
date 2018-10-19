clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example for integrating EDPP screening rule with SLEP to
% solve the Lasso problem at a given sequence of parameter values:
%
%  min  1/2 || X * beta - y||^2 + lambda * ||beta||_1
%  s.t.  beta >= 0
%
% Author: Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate the data                     
m=250;  n=10000;     % The data matrix is of size m x n
X = normrnd(0,1,[m,n]);
y = normrnd(0,1,[m,1]);

%% set up the solver
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

% choose the model
opts.fName = 'nnLeastR';  % compute a sequence of nonnegative lasso problems

%% set the regularization paramter values
% if the parameter values are the ratios of lambda/lambda_max; if use the 
% absolute value, please set opts.rFlag = 0
%
ub = 1; % the upper bound of the paramter values
lb = 0.01; % the lower bound of the parameter values
npar = 100; % the number of parameter values
delta_lambda = (log(ub) - log(lb))/(npar-1);
lambda = exp(log(lb):delta_lambda:log(ub)); % the paramter sequence

%% solve the nonnegative Lasso problems along the sequence of parameter values

tic
[Sol, ind_zf] = DPC_nnLasso(X, y, lambda, opts);
toc    

