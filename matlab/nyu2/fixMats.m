pathNYU2 = '/data/vision/fisher/data1/nyu_depth_v2/';

% fix for collected mat files that will change mfRs such that there are
% rotations only for MFs with data and fix the masks which were off as
% welljstraubjst

% load([pathNYU2 'nyu_depth_v2_labeled.mat'],'scenes');
% load([pathNYU2 'nyu_depth_v2_mmf_v1.1.mat'])

for i=1:length(scenes)
    n_mmfs = size(unique(mmfs(:,:,i)),1)-1;
    n_Rs = size(mfRs{i},2)/3;
    counts = sum( bsxfun(@eq, reshape(mmfs(:,:,i),[],1), [0:n_Rs]) )';
%         counts = sum( bsxfun(@eq, reshape(mmfs(:,:,i),[],1), unique(reshape(mmfs(:,:,i),[],1))') )';
    if n_mmfs ~= n_Rs 
        disp(sprintf('%d \t arg %d vs %d ',i,n_mmfs,n_Rs));
        counts'
        Rs = [];
        % dont count id =0 because that are invalid points
        for j = [1:length(counts)-1]
            if counts(j+1) > 0 
                Rs=[Rs, mfRs{i}(:,(j-1)*3+1:j*3)];
            end
        end
        mfRs{i} = Rs;
    end
    if sum(masks(:,:,i) ~= (mmfs(:,:,i)~=0))
        masks(:,:,i) = (mmfs(:,:,i)~=0);
    end
end

% save([pathNYU2 'nyu_depth_v2_mmf_T120_S0.07_K06_T80_origMMFupsampleML_fixedRsMasks.mat'],'scenes','mmfs','mfss','logLikeNormals','mfRs','masks','-v7.3');


load([pathNYU2 'nyu_depth_v2_normals.mat'],'normals');
save([pathNYU2 'nyu_depth_v2_normals.mat'],'scenes','masks','normals','-v7.3');