function compute_normals_from_file(filename)
profile off;profile on
W=2; % filter radius
R=imread(filename);
R=double(R);
% f = 400.0; % focal length of asus kinect
f = 525.0; % focal length of asus kinect
invf = 1.0/f; % inverse focal length of asus kinect

[U,V]=meshgrid((1:size(R,2))-size(R,2)/2,(1:size(R,1))-size(R,1)/2);
z = R*0.001;
x = z.*U*invf;
y = z.*V*invf;
tic
n=estimate_normals(x,y,z,W);
toc
u=[];
u(:,:,1)=n.nx;
u(:,:,2)=n.ny;
u(:,:,3)=n.nz;
mask=~(sum(isnan(u),3)>0 | z==0);
for i=1:3
    % replace nan values
    v=u(:,:,i);v(~mask)=randn(size(v(~mask)));
    u(:,:,i)=v;
end
u=project_onto_s2(u);
opts=[];opts.lambda=1;
opts.verbose=false;opts.OUTER_ITER=10;

% Perform denoising
tic
[regularized_u,stats]=al_s2(u,opts);
toc
u1=[];
u2=[];
u3=[];
    
for i = 1:3
    vtmp=stats.v(:,:,i);
    utmp=u(:,:,i);
    u3(:,i)=vtmp(:);
    u2(:,i)=vtmp(mask);
    u1(:,i)=utmp(mask);
end
out_filename=[filename,'.normals.mat'];
%vr=5e3;bias=0.1;R4=[];R1=R/vr-bias;R1(region1)=1;R1(region3)=0;R1(region2)=0;R4=R1;R1=R/vr-bias;R1(region1)=0;;R1(region3)=0;R1(region2)=1;R4(:,:,2)=R1;R1=R/vr-bias;R1(region3)=1;R4(:,:,3)=R1;imshow(R4,[]);
%nl3=[];nl2=[];nl1=[];for i=1:3;vtmp=stats.v(:,:,i);nl3(i)=mean(mean(vtmp(region3)));nl1(i)=mean(mean(vtmp(region1)));nl2(i)=mean(mean(vtmp(region2)));end
save(out_filename,'u','regularized_u','mask');
end
