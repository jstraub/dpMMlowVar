path = '/data/vision/fisher/data1/nyu_depth_v2/'

%load([path 'nyu_depth_v2_labeled.mat']);

N=size(scenes,1)
saveImgs = true

fid = fopen([path 'index.txt'],'w');
for i = 1:N
    rgb=images(:,:,:,i);
    %d=depths(:,:,i);
    
    d=rawDepths(:,:,i);
%     maxD = max(d(:));
%     minD = min(d(:));
%     dImg = (d-minD)/(maxD-minD);
    
    name = [scenes{i} '_' int2str(i) ];
    d=uint16(floor(d*1000));
    
    if saveImgs
        imwrite(rgb,[path 'extracted/' name '_rgb.png']);
        imwrite(d,[path 'extracted/' name '_d.png']);
    end
    disp(['image: ' name ]);
    
    fprintf(fid,[ name '\n']);
    
end
fclose(fid);