% Generate data as NxD in the Matlab style, but will tranpose it for using DDP
% Data is three well-separated Gaussian blobs that get moved uniformly.
N = 60;
X1 = vertcat(mvnrnd([-10,-10], eye(2), N/3), ...
             mvnrnd([10,10], eye(2), N/3), ...
             mvnrnd([0,0], eye(2), N/3));
X2 = X1 + repmat([1, 1], [N,1]);
y = vertcat(zeros(N/3,1), ones(N/3,1), 2*ones(N/3,1));
colors = [1 0 0; 0 1 0; 0 0 1];

% DDP parameters
lambda = 30;
Q = 1;
tau = 1;
T = 100;
doRevive = 0;

% Call DDP;
% NOTE: must ALWAYS call 'init', then 0 or more 'step' calls, then % 'close'.
% This stores state between calls, so a process may only have one of % them.
[z, ctr, cost, dev, wts, age] = mex_ddp('init', X1', lambda, Q, tau, T);
[z2, ctr2, cost2, dev2, wts2, age2] = mex_ddp('step', X2', doRevive);
mex_ddp('close');

% visualize
subplot(2,2,1); scatter(X1(:,1), X1(:,2), 10, colors(y+1,:), 'filled');
title('Time=0, True Clusters');

subplot(2,2,2); scatter(X2(:,1), X2(:,2), 10, colors(y+1,:), 'filled');
title('Time=1, True Clusters');

subplot(2,2,3); scatter(X1(:,1), X1(:,2), 10, colors(z'+1,:), 'filled');
title('Time=0, Estimated Clusters');

subplot(2,2,4); scatter(X2(:,1), X2(:,2), 10, colors(z2'+1,:), 'filled');
title('Time=1, Estimated Clusters');
