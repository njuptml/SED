%pearson
clear
clc
X=load('select_1_90.mat');
t=load('label1.mat');
X=X.select;
y=t.label;
C=[y,X];

%C(:,1) = sum(C,2);   % Introduce correlation.
[r,p] = corrcoef(C);  % Compute sample correlation and p-values.
%[i,j] = find(p<0.05);  % Find significant correlations.
                 