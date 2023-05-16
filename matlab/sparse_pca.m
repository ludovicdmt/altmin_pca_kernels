function [W, C] = sparse_pca(X, n_comps, alpha, tolerance)
    if nargin < 4
        tolerance = 0.001;
    end

    cost_func_w = @(predW, c, Xfit) compute_cost_func_w(predW, c, Xfit, alpha);
    cost_func_c = @(predC, w, Xfit) compute_cost_func_c(predC, w, Xfit, alpha);

    [W, C] = fit(X, n_comps, tolerance, cost_func_w, cost_func_c);
end

function cost = compute_cost_func_w(predW, c, Xfit, alpha)
    predX = predW * c;
    pred_diff = predX - Xfit;
    err = (norm(pred_diff, 'fro')) ^ 2;
    l1_reg = alpha * sum(norm(predW, 1)) + alpha * sum(norm(c, 1));
    cost = err + l1_reg;
end

function cost = compute_cost_func_c(predC, w, Xfit, alpha)
    predX = w * predC;
    pred_diff = predX - Xfit;
    err = (norm(pred_diff, 'fro')) ^ 2;
    l1_reg = alpha * sum(norm(w, 1)) + alpha * sum(norm(predC, 1));
    cost = err + l1_reg;
end