function ratio=initFactor(x_norm, Ax , y, z, funName, rsL2, x_2norm)
% 
%% function initFactor
%     compute the an optimal constant factor for the initialization
%
%
% Input parameters:
% x_norm-      the norm of the starting point
% Ax-          A*x, with x being the initialization point
% y-           the response matrix
% z-           the regularization parameter or the ball
% funName-     the name of the function
%
% Output parameter:
% ratio-       the computed optimal initialization point is ratio*x
%
%% 
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last revised on August 2, 2009.
%
%% License
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Copyright (C) 2009 - 2012 Jun Liu and Jieping Ye 
%
%%

switch(funName)
    case 'LeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'LeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        %fprintf('\n ratio=%e',ratio);
        
    case 'glLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'mcLeastR'
        ratio=  (Ax(:)'*y(:) - z * x_norm) / norm(Ax,'fro')^2;
        %fprintf('\n ratio=%e',ratio);
        
    case 'mtLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'nnLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        ratio=max(0,ratio);
        
    case 'nnLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);

        if ratio_optimal<0
            ratio=0;
        elseif ratio_optimal<=ratio_max
            ratio  =  ratio_optimal;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'mcLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax(:)'*y(:) / (norm(Ax'*Ax,'fro')^2);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        
    otherwise
        fprintf('\n The specified funName is not supprted');
end
