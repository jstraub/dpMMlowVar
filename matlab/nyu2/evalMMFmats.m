

reload=false;
if reload
    clear all;
    pathNYU2 = '/data/vision/fisher/data1/nyu_depth_v2/';
%         load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.085_K06.mat']);
    %load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.1_K06.mat'])
    load([pathNYU2 'nyu_depth_v2_mmf_T120_S0.07_K06_T80_origMMF.mat'])
    load([pathNYU2 'nyu_depth_v2_labeled.mat'],'images');
    gt = int32(load('/data/vision/fisher/data1/nyu_depth_v2/mfCountGroundtruth.mat','-ASCII'));
end

axisColor = [1,0,0; 1,0.1,0.1; 0,1,0; 0.1,1,0.1; 0,0,1; 0.1,0.1,1];
plot=true;

thresh = 0.15
correct = zeros(1,1449)

figure()
for i=1:length(scenes)%[935,17,576,358,1314,365]%1:length(scenes)    
    % soimple way without noise removal
    nMMFs = int32(max(max(mmfs(:,:,i))));
    counts = zeros(nMMFs,1);
    for k=1:nMMFs
        counts(k) = sum(sum(mmfs(:,:,i)==k));
    end
    counts = counts/sum(sum(mmfs(:,:,i)>0));
    nMMFs = sum(counts > thresh);
    
    if gt(i,1) ~= nMMFs 
        disp(sprintf('@i=%d: GT=%d vs. Inferred=%d',i,gt(i,1),nMMFs))
        if plot
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
        correct(i) = 0;
    else
        disp(sprintf('@i=%d: Inferred=%d matches GT',i,nMMFs))
        correct(i) = 1;
    end
end

sum(correct)/1449.