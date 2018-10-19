%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ECFP_12_1024
% This is a demo for integrating EDPP screening rule with SLEP to
% solve the Lasso problem at a given sequence of parameter values:
%
% min  1/2 || X * beta - y||^2 + lambda * ||beta||_1
%
% Author: Benli
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set your input                    
clear all;
gpcr_name = 'P08908'   % Please input gpcr_name in the folder "data", eg: 'P08908', 'P28335', 'P35372'

gpcr_length = 1024     % Please input length of gpcr, eg:1024, 5120, 10240, 51200, 102400

gpcr_radius = 6        % Please input radius of gpcr, eg:6, 4, 2
cd DPC\
%% set the data matrix  
File_name =strcat(gpcr_name,'_ECFP', num2str(gpcr_radius*2),'_', num2str(gpcr_length));
[status, cmdout] = system(strcat('python Gen_ECFPs.py', 32 , gpcr_name, 32, num2str(gpcr_length), 32 , num2str(gpcr_radius)))

load(strcat(File_name, '.mat'));
X=double(X);
X1 = zscore(X);
y = csvread(strcat('../data/',gpcr_name,'/Response.csv'));
load(strcat('number', num2str(gpcr_length), '.mat'));
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

ub = 0.5; % upper bound of the parameter values
lb = 0.05; % lower bound of the parameter values
npar = 100; % number of parameter values
delta_lambda = (ub - lb)/(npar-1);
lambda=lb:delta_lambda:ub; % the parameter sequence

%% solve the Lasso problems along the sequence of parameter values

tic
[Sol, ind_zf] = EDPP_Lasso(X1, y, lambda, opts);
T=ind_zf(:,1);
Xr = X1(:,~T);
ind=sum(~ind_zf,1);
intx=sum(~ind_zf,2);
final=[fn',intx];
[~,I]=sort(-final(:,2));
final=final(I,:);
%intx=100-intx;
toc

%% select key features
t=1;
for i=1:300
    sel(t)=final(i,1);
    t=t+1;
end

for j=1:length(X)
    for k=1:(t-1)
        select(j,k)=X(j,sel(k));
    end
end

%% Use DNN to predict Response
csvwrite(strcat(File_name, '_Top300','.csv'),select)
[status, cmdout] = system(strcat('python Prepare_DNN_Data.py', 32 , gpcr_name, 32, num2str(gpcr_length), 32 , num2str(gpcr_radius)))
cd ..
cd DeepNeuralNet-QSAR\
system(strcat('python Prepare_DNN_Data.py', 32 , gpcr_name, 32, num2str(gpcr_length), 32 , num2str(gpcr_radius)))
[status, cmdout] = system(strcat('python DeepNeuralNetTrain.py', 32 , '--seed=0', 32, '--hid=4000', 32 , '--hid=2000', 32, '--hid=1000', 32, '--hid=1000', 32, '--dropout=0_0.25_0.25_0.25_0.1',32, '--epochs=2',32,strcat('--data=',gpcr_name),32,strcat('models/',gpcr_name)))
[status, cmdout] = system(strcat('python DeepNeuralNetPredict.py', 32 , '--seed=0', 32, '--label=1', 32 , '--rep=10',32, strcat('--data=',gpcr_name),32, strcat('--model=models/',gpcr_name),32,strcat('--result=predictions/',gpcr_name)))
cd ..