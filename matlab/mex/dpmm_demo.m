% Generate data as NxD in the Matlab style, but will tranpose it for DPMM call.
% Just a few well-separated Gaussian blobs.
N = 60;
X1 = vertcat(mvnrnd([-10,-10], eye(2), N/3), ...
             mvnrnd([10,10], eye(2), N/3), ...
             mvnrnd([0,0], eye(2), N/3));
y = vertcat(zeros(N/3,1), ones(N/3,1), 2*ones(N/3,1));
colors = [1 0 0; 0 1 0; 0 0 1];

% set DPMM parameters
alpha = 1;
initK = 0;
T = 100;
lambda = 30;

% call
[z, ctr, cost, dev] = mex_dpmm(X1', alpha, initK, T, lambda);

% visualize
subplot(2,1,1); scatter(X1(:,1), X1(:,2), 10, colors(y+1,:), 'filled');
title('True Clusters');

subplot(2,1,2); scatter(X1(:,1), X1(:,2), 10, colors(z'+1,:), 'filled');
title('Estimated Clusters');
