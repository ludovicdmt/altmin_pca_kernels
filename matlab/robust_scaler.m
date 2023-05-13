function scaled_data = robust_scaler(X, quant_range)
% Scales the data in X by removing the median and scaling according to
% the quantile range. The default quantile range is IQR (Interquartile Range).
% 
% Inputs:
%   X: samples x features matrix of data
%   quant_range: scalar value specifying the quantile range to use
% 
% Outputs:
%   scaled_data: samples x features matrix of scaled data

if nargin < 2
    quant_range = 0.5; % default to IQR
end

med = median(X);
q1 = quantile(X, 0.25);
q3 = quantile(X, 0.75);

% Calculate the IQR
IQR = q3 - q1;

% Compute the lower and upper bounds for scaling
lower_bound = med - quant_range * IQR;
upper_bound = med + quant_range * IQR;

% Scale the data between the lower and upper bounds
scaled_data = (X - med) ./ (upper_bound - lower_bound);
end