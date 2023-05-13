function [W, C] = quadratically_regularized_pca(X, n_comps, alpha, tolerance)
    if nargin < 4
        tolerance = 0.001;
    end
    if nargin < 3
        alpha = 1.0;
    end

    cost_func_w = @(predW, c, Xfit) compute_cost_func_w(predW, c, Xfit, alpha);
    cost_func_c = @(predC, w, Xfit) compute_cost_func_c(predC, w, Xfit, alpha);

    [W, C] = fit(X, n_comps, tolerance, cost_func_w, cost_func_c);
end

function cost = compute_cost_func_w(predW, c, Xfit, alpha)
    predX = predW * c;
    pred_diff = predX - Xfit;
    err = norm(pred_diff, 'fro')^2;
    l2_reg = alpha * sum(norm(predW, 'fro')^2) + alpha * sum(norm(c, 'fro')^2);
    cost = err + l2_reg;
end

function cost = compute_cost_func_c(predC, w, Xfit, alpha)
    predX = w * predC;
    pred_diff = predX - Xfit;
    err = norm(pred_diff, 'fro')^2;
    l2_reg = alpha * sum(norm(w, 'fro')^2) + alpha * sum(norm(predC, 'fro')^2);
    cost = err + l2_reg;
end