pathNYU2 = '/data/vision/fisher/data1/nyu_depth_v2/';

% load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.085_K06.mat'])
% load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.1_K06.mat'])
load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.07_K06_T80_origMMF.mat'])
load([pathNYU2 'nyu_depth_v2_labeled.mat'],'images')

axisColor = [1,0,0; 1,0.1,0.1; 0,1,0; 0.1,1,0.1; 0,0,1; 0.1,0.1,1]

figure()
for i=1:length(scenes)%[935,17,576,358,1314,365]%1:length(scenes)
    subplot(2,2,1);
    imshow(images(:,:,:,i));
    subplot(2,2,2);
    imshow(label2rgb(mmfs(:,:,i),lines));
    subplot(2,2,3);
    imshow(label2rgb(mfss(:,:,i),axisColor));
    subplot(2,2,4);
    imshow(logLikeNormals(:,:,i)); colormap hot;
    input('enter to continue')
end

