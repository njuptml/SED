clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example for integrating TLFre screening rule with SLEP to
% solve the sparse-group Lasso (SGL) problem at given sequences of parameter
% values. The original sparse-group Lasso takes the form of
%
%  min  1/2 || X * beta - y||^2 + lambda1 * sum_i w_i * ||beta_i||_2 + lambda2 * ||beta||_1            
%
% Let 
%
%    alpha = lambda1/lambda2,   lambda = lambda2,
%
% we consider the following reparameterized SGL:
%
%  min  1/2 || X * beta - y||^2 + lambda * alpha * sum_i w_i * ||beta_i||_2 + lambda * ||beta||_1      
%
%  The group information is contained in
%  opts.ind, which is a 3 x nodes matrix, where nodes denotes the number of
%  nodes of the tree.
%  opts.ind(1,:) contains the starting index
%  opts.ind(2,:) contains the ending index
%  opts.ind(3,:) contains the corresponding weight (w_j)
%
% Related papers
%
% [1] Jie Wang and Jieping Ye, Two-Layer Feature Reduction for Sparse-Group
%     Lasso via Decomposition of Convex Sets, NIPS 2014
%
% Author: Jie Wang (jiewangustc@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate synthetic data
m=250;  n=10000;     % The data matrix is of size m x n
X = normrnd(0,1,[m,n]);
y = normrnd(0,1,[m,1]);
[num_samples,num_features] = size(X);
group = 0:10:n; % the group index; the ith group include ind(i)+1:ind(i+1) features

% --------------------- set up the group information --------------
ng = length(group)-1;          % the number of groups
sg = diff(group);              % the size of each group, i.e., number of features contained in each group
ind = zeros(3,ng);
ind(1,:) = group(1:end-1)+1;
ind(2,:) = group(2:end);
ind(3,:) = sqrt(sg);
group_ind = zeros(1,size(X,2)); % the group index of each feature
for i = 1 : ng
    group_ind(ind(1,i):ind(2,i)) = i;
end

%% set up the regularization paramters alpha and lambda_L
% --------------------- set alpha ------------------------------
angle = [5 15:15:75 85];
alpha = tan(angle/360*(2*pi));
% --------------------- set lambda -------------------------------
% if the parameter values are the ratios of lambda/lambda_max; if use the 
% absolute value, please set opts.rFlag = 0
%
ub = 1; % the upper bound of the paramter values
lb = 0.01; % the lower bound of the parameter values
npar = 100; % the number of parameter values
delta_lambda = (log(ub) - log(lb))/(npar-1);
lambda = exp(log(lb):delta_lambda:log(ub)); % the paramter sequence

%----------------------- Set optional items -----------------------
opts=[];

% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=1000;   % maximum number of iterations

% regularization
opts.rFlag=1;       % use ratio

% Normalizationl
opts.nFlag=0;       % without normalization

% Group Property
opts.ind=ind;
opts.gind = group_ind;

%% solve the SGL problems for all the given parameter value pairs

[Sol, ind_zf1, ind_zf2] = TLFre_SGL(X, y, alpha, lambda, opts);






