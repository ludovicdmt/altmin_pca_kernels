function [W, C] = fit(X, n_comps, tolerance, cost_func_w, cost_func_c)
n_observations = size(X, 1);
n_features = size(X, 2);

W = zeros(n_observations, n_comps);
C = zeros(n_comps, n_features);

Xfit = X;

% Fit one component at a time
for comp_idx = 1:n_comps
    
    % Initialize w and c
    w = randn(n_observations, 1);
    c = randn(1, n_features);
    
    iter = 1;
    last_err = 0;
    err = norm((w * c) - Xfit, 'fro');
    
    % while SC not satisfied, do
    while abs(err - last_err) > tolerance
        last_err = err;
        
        % Optimize w - one step
        options = optimoptions('fminunc', 'Display', 'off', 'MaxIterations', 1);
        res = fminunc(@(x) cost_func_w(x, c, Xfit), w, options);
        w = reshape(res, [n_observations, 1]);
        
        % Optimize c - one step
        options = optimoptions('fminunc', 'Display', 'off', 'MaxIterations', 1);
        res = fminunc(@(x) cost_func_c(x, w, Xfit), c, options);
        c = reshape(res, [1, n_features]);
        
        % update error and count
        pred_diff = w * c - Xfit;
        err = norm(pred_diff, 'fro');
        iter = iter + 1;                
    end
    fprintf('PC%d: iterations=%d, error=%f\n', comp_idx, iter, err);
    Xfit = Xfit - (w * c);
    W(:, comp_idx) = w(:, 1);
    C(comp_idx, :) = c(1, :);

end