
load('all_bursts.mat');

ncomps = 20;

% Scale the data using RobustScaler
burst_std = robust_scaler(all_bursts);

% Compute covariance matrix and its eigenvectors and eigenvalues
covariance_matrix = cov(burst_std);
[eigen_vectors, eigen_values] = eig(covariance_matrix);

% Sort eigenvectors and eigenvalues in decreasing order of eigenvalues
[eigen_values, indices] = sort(diag(eigen_values), 'descend');
eigen_vectors = eigen_vectors(:, indices);
eigen_values=eigen_values(1:ncomps);
eigen_vectors=eigen_vectors(:,1:ncomps);

% Compute principal components
pcs = burst_std * eigen_vectors;

% Compute explained variance ratio and cumulative variance ratio
explained_variance_ratio = eigen_values / sum(eigen_values);
cumulative_variance_ratio = cumsum(explained_variance_ratio);

% Plot variance ratios for each principal component
PC = 1:ncomps;
figure('Color','w');
subplot(1,2,1)
bar(PC, explained_variance_ratio);
xlabel('Principal Components')
ylabel('Variance %')
xticks(PC)

subplot(1,2,2)
plot(PC, cumulative_variance_ratio);
xlabel('Principal Components')
ylabel('Cumulative Variance %')

pcs_to_analyze = 20;
disp(cumulative_variance_ratio(8));



[W,C]=pca(burst_std, ncomps);

% Define cool_colors using cool colormap
cool_colors = cool(ncomps);

% Create figure and plot each PC and its corresponding component
figure('Color','w', 'Position',[0 0 1200 800]);
for i = 1:ncomps
    subplot(8,5,i);
    plot(C(i,:), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

for i = 1:ncomps
    subplot(8,5,ncomps+i);
    plot(eigen_vectors(:,i), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

% Compute correlation matrix and plot
corrmat = zeros(ncomps);
for i = 1:ncomps
    for j = 1:ncomps
        [r, ~] = corrcoef(eigen_vectors(:,i), C(j,:));
        corrmat(i,j) = abs(r(1,2));
    end
end

figure('Color','w');
ax = imagesc(corrmat);
set(gca,'Ydir','normal');
set(ax, 'XData', [1, pcs_to_analyze], 'YData', [pcs_to_analyze, 1]);
axis equal tight
colormap(ax.Parent, jet);
colorbar;
xlabel('classic PCA');
ylabel('optimized PCA');





[W,C]=quadratically_regularized_pca(burst_std, ncomps, 50);

% Define cool_colors using cool colormap
cool_colors = cool(ncomps);

% Create figure and plot each PC and its corresponding component
figure('Color','w', 'Position',[0 0 1200 800]);
for i = 1:ncomps
    subplot(8,5,i);
    plot(C(i,:), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

for i = 1:ncomps
    subplot(8,5,ncomps+i);
    plot(eigen_vectors(:,i), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

% Compute correlation matrix and plot
corrmat = zeros(ncomps);
for i = 1:ncomps
    for j = 1:ncomps
        [r, ~] = corrcoef(eigen_vectors(:,i), C(j,:));
        corrmat(i,j) = abs(r(1,2));
    end
end

figure('Color','w');
ax = imagesc(corrmat);
set(gca,'Ydir','normal');
set(ax, 'XData', [1, pcs_to_analyze], 'YData', [pcs_to_analyze, 1]);
axis equal tight
colormap(ax.Parent, jet);
colorbar;
xlabel('classic PCA');
ylabel('optimized PCA');


[W,C]=sparse_pca(burst_std, ncomps, 20);

% Define cool_colors using cool colormap
cool_colors = cool(ncomps);

% Create figure and plot each PC and its corresponding component
figure('Color','w', 'Position',[0 0 1200 800]);
for i = 1:ncomps
    subplot(8,5,i);
    plot(C(i,:), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

for i = 1:ncomps
    subplot(8,5,ncomps+i);
    plot(eigen_vectors(:,i), 'color', cool_colors(i,:));
    title(['PC ', num2str(i)]);
end

% Compute correlation matrix and plot
corrmat = zeros(ncomps);
for i = 1:ncomps
    for j = 1:ncomps
        [r, ~] = corrcoef(eigen_vectors(:,i), C(j,:));
        corrmat(i,j) = abs(r(1,2));
    end
end

figure('Color','w');
ax = imagesc(corrmat);
set(gca,'Ydir','normal');
set(ax, 'XData', [1, pcs_to_analyze], 'YData', [pcs_to_analyze, 1]);
axis equal tight
colormap(ax.Parent, jet);
colorbar;
xlabel('classic PCA');
ylabel('optimized PCA');
