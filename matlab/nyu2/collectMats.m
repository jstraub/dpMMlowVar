pathNYU2 = '/data/vision/fisher/data1/nyu_depth_v2/';
pathResults = '/data/vision/scratch/fisher/jstraub/mmf/results/multiFromFile/';
pathResults = '/data/vision/fisher/expres1/jstraub/mmf/mmf_nyu/results/multiFromFile';

load([pathNYU2 'nyu_depth_v2_labeled.mat'],'scenes');
N = length(scenes);
normals = single(zeros(480,640,3,N));
logLikeNormals = single(zeros(480,640,N));
mmfs = uint8(zeros(480,640,N));
mfss = uint8(zeros(480,640,N));
mfRs = cell(N,1);
masks = false(480,640,N);

for i=1:N
%     name = [scenes{i} '_' int2str(i) '_6_0.1_results.mat'];
%    name = [scenes{i} '_' int2str(i) '_results.mat'];
    name = [scenes{i} '_' int2str(i) '_6_0.07_80_80_results.mat'];
    if exist([pathResults name])
        a = load([pathResults name]);
        mmfs(:,:,i) = a.mmf;
        mfss(:,:,i) = a.mfs;
        logLikeNormals(:,:,i) = a.logLike;
        mfRs{i} = a.Rs;
        masks(:,:,i) = a.mask;
        normals(:,:,:,i) = a.normal;
        disp(['incorporate '  name])
    else
        warning(['skipping '  name])
    end
end

% save([pathNYU2 'nyu_depth_v2_normals.mat'],'scenes','masks','normals','-v7.3');
save([pathNYU2 'nyu_depth_v2_mmf_T120_S0.07_K06_T80_origMMF.mat'],'scenes','mmfs','mfss','logLikeNormals','mfRs','masks','-v7.3');
