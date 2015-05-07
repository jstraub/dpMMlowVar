pathNYU2 = './';


% load([pathNYU2 'nyu_depth_v2_mmf_v1.0.mat'])
% load([pathNYU2 'nyu_depth_v2_labeled.mat'],'images')

axisColor = [1,0,0; 1,0.1,0.1; 0,1,0; 0.1,1,0.1; 0,0,1; 0.1,0.1,1]

for i=1:length(scenes)
    n_mmfs = size(unique(mmfs(:,:,i)),1)-1;
    n_Rs = size(mfRs{i},2)/3;
    counts = sum( bsxfun(@eq, reshape(mmfs(:,:,i),[],1), unique(reshape(mmfs(:,:,i),[],1))') )';
    if n_mmfs ~= n_Rs 
        disp(sprintf('%d \t arg %d vs %d ',i,n_mmfs,n_Rs));
        counts'
         unique(reshape(mmfs(:,:,i),[],1))'
    end
end


figure()
for i=1:length(scenes)
    masks(:,:,i) = (logLikeNormals(:,:,i)== 0);
    subplot(2,3,1);
    imshow(bsxfun(@times,double(images(:,:,:,i)),masks(:,:,i)));
    subplot(2,3,2);
    imshow(label2rgb(mmfs(:,:,i),lines));
    subplot(2,3,3);
    imshow(label2rgb(mfss(:,:,i),axisColor));
    subplot(2,3,4);
    imshow(bsxfun(@times,logLikeNormals(:,:,i),masks(:,:,i))); colormap hot;
    subplot(2,3,5);
    imshow(masks(:,:,i));
    input('enter to continue')
end
