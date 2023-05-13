function [W, C] = fit_pca(X, n_comps, tolerance)
    if nargin < 3
        tolerance = 0.001;
    end

    % Define cost functions
    cost_func_w = @(predW, c, Xfit) norm((predW * c) - Xfit, 'fro')^2;
    cost_func_c = @(predC, w, Xfit) norm((w * predC) - Xfit, 'fro')^2;

    % Call _fit function
    [W, C] = fit(X, n_comps, tolerance, cost_func_w, cost_func_c);

end