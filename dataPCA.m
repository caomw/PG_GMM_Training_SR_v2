function  [X_pca, V_pca]= dataPCA (X)

% PCA dimensionality reduction
C = double(X * X');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix 
D = cumsum(D) / sum(D);
k = find(D >= 1e-3, 1); % ignore 0.1% energy
V_pca = V(:, k:end); % choose the largest eigenvectors' projection
X_pca = V_pca' * X;

end