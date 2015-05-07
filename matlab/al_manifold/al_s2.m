% Implement TV denoising of SE(3) elements on a cartesian 2D grid
%
% usage: [u,stats]=al_se3(I,options), where
%
% I - an MxNx9 image
% options - an options structure, with the optional fields:
%   'lambda' (defaults to 15) - the strength of the fidelity term
%   'iter_q_1' (defaults to 2) - the number of inner Gauss-Seidel iterations
%   for updating q
%   'iter_q_2' (defaults to 60) - the number of IRLS iterations for updating q.
%   In practice if the residual is low enough, the IRLS iterations are terminated.
%   'OUTER_ITER' (defaults to 100) - the number of outer iterations updating \mu, the Lagrange multipliers.
%   'INNER_ITER' (defaults to 1) - the number of inner iterations updating q,u
%   'beta' (defaults to 1000) - the Spatial-Chromal ratio
%   'r_updates' (defaults to 3) - the number of updates to r. Scale factor
%   used is 2.
%   'initial_r' (defaults to 1) - the initial r value
%
% and the outputs are:
%
% u - the result image
% stats - a structure with various residual measurements
%
% Guy Rosman (kenwash@gmail.com), 2011
function [u,stats]=al_s2(I,options)
defaults=struct('lambda',15,'blur_operator',[],'save_video',[],'pose1',[],'pose2',[],...
    'iter_q_1',2,'iter_q_2',2,'OUTER_ITER',150,'INNER_ITER',1,'beta',4000,'r_updates',2,...
    'initial_r',2,'initial_rz',1,'initial_r2',2,'I_ideal',[],'compute_metric',false,'padding',0,'reduce_linear_diffusion',true,...
    'use_rre',false,'no_gradient_coupling',false,'scheme','backward_forward','fidelity','L2','valid_map',[],'stability_factor',1e0,'fidelity_weight',[],'IRLS_EPS',1e-2,'IRLS_ITER',0,'diffusivity',[]);
color_dim=size(I,3);
options=incorporate_defaults(options,defaults);
if (~isempty(options.save_video))
    figure(1)
    mov=avifile(options.save_video,'fps',5);
end
if (options.padding>0)
    I=padarray(I,options.padding*[1 1 0],'replicate','both');
    options.I_ideal=padarray(options.I_ideal,options.padding*[1 1 0],'replicate','both');
end
r=options.initial_r;
rz=options.initial_rz;
r2=options.initial_r2;
stability_factor=options.stability_factor;
dx=1;dy=1;
gu=zeros([size(I,1),size(I,2),2*color_dim]);
switch lower(options.fidelity)
    case 'l1'
        z=zeros(size(I));
        muz=zeros(size(z));
end
try
    u=options.u_init;
    for i  = 1:color_dim
        gu(:,:,(i-1)*2+1)=Dmx(u(:,:,i),dx);
        gu(:,:,(i-1)*2+2)=Dmy(u(:,:,i),dy);
    end
    q=gu;
catch
    q=zeros([size(I,1),size(I,2),2*color_dim]);
    u=I;
    for i  = 1:color_dim
        q(:,:,(i-1)*2+1)=Dmx(u(:,:,i),dx);
        q(:,:,(i-1)*2+2)=Dmy(u(:,:,i),dy);
    end
end

difmat=make_imfilter_mat([0 1 0; 1 -4 1; 0 1 0],size(I(:,:,1)),'replicate');
idmat=speye(numel(I(:,:,1)));
t0=cputime;
if (isempty(options.fidelity_weight))
options.fidelity_weight=ones(size(I(:,:,1)));
end

%initialize grad u,mu,q
for i  = 1:color_dim
    gu(:,:,(i-1)*2+1)=Dmx(u(:,:,i),dx);
    gu(:,:,(i-1)*2+2)=Dmy(u(:,:,i),dy);
end
mu=zeros(size(I,1),size(I,2),color_dim*2);
mu2=zeros(size(I,1),size(I,2),color_dim); % for SE3 constrain
v=project_onto_s2(u);

if isempty(options.blur_operator)
    blur_op=1;
else
    blur_op=options.blur_operator;
end
% compute the fourier transform of the image
% fI=imfilter(I,blur_op,bndCond);
% fI=zeros(size(I));
% for i = 1:color_dim
%     fI(:,:,i)=fft2(I(:,:,i));
% end
% compute the fourier transform of the Laplacian
% zlap=zeros([3,3t]);
% zlap(2,2)=1;
% % zlap=fftshift(zlap);
% lap=(Dpx(Dmx(zlap))+Dpy(Dmy(zlap)));
% flap=ifftshift(lap);
% flap=fft2(flap);


  
% iteration residual vectors, for profiling
res=zeros(options.OUTER_ITER*options.INNER_ITER,1);
stats.gaps=zeros(options.OUTER_ITER*options.INNER_ITER,1);
stats.gaps2=zeros(options.OUTER_ITER*options.INNER_ITER,1);
res2=zeros(options.OUTER_ITER*options.INNER_ITER,1);
stats.orth_dev=[];
ires=zeros(options.OUTER_ITER*options.INNER_ITER,1);
ts=ires;
stats.psnr=ires;
stats.ressss=[];stats.nresss=[];
res_cnt=0;
res2_cnt=0;
ires_cnt=0;
if (~isempty(options.I_ideal))
    res2_cnt=res2_cnt+1;
    stats.psnr(res2_cnt)=-10*log10(norm(options.I_ideal(:)-u(:))^2/numel(u));
    ts(res2_cnt)=cputime-t0;
end
if (options.use_rre)
    iterants_u=[];
    iterants_q=[];
    iterants_mu=[];
end
t1=cputime;
tic;
% the main loop
IRLS_weights=ones(size(u(:,:,1)));
for k=1:options.r_updates

    for j=1:options.OUTER_ITER
        if (options.IRLS_ITER>0 && mod(j,options.IRLS_ITER)==0)
            IRLS_weights=(sum((u-I).^2,3)+options.IRLS_EPS).^(-0.5);
        end
  LHSmat=-difmat*r+(stability_factor+r2)*idmat+options.lambda*spdiags(options.fidelity_weight(:).*IRLS_weights(:),0,idmat);
    LHS_L=tril(LHSmat);
    LHS_U=LHSmat-LHS_L;
        u1=u;
        for i = 1:options.INNER_ITER
            u0=u;
            % possibly update z if we're using an L1 fidelity
            switch lower(options.fidelity)
                case 'l1'
                    for d=1:color_dim
                        %                             if (isempty(options.valid_map))
                        w=imfilter(u(:,:,d),blur_op,'circular')-muz(:,:,d)/rz;
                        dw=w-I(:,:,d);
                        z(:,:,d)=I(:,:,d)+max(0,1-(options.lambda/rz)./(abs(dw)+1e-15)).*dw;
                        %                             else
                        %                             w=imfilter(u(:,:,d),blur_op,'circular')-muz(:,:,d)./reff;
                        %                             dw=w-I(:,:,d);
                        %                             z(:,:,d)=I(:,:,d)+max(0,1-(options.lambda/rz)./(abs(dw)+1e-15)).*dw;
                        %                             end
                    end
                    
            end
            % update u, given q,mu,
            for d = 1:color_dim
                muc=mu(:,:,(1:2)+(d-1)*2);
                mu2c=mu2(:,:,d);
                Ic=I(:,:,d);
                uc=u(:,:,d);
                vc=v(:,:,d);
%                 fIc=fI(:,:,d);
                qc=q(:,:,(1:2)+(d-1)*2);
                divq=Dpx(qc(:,:,1),dx);
                divq=divq+Dpy(qc(:,:,2),dy);
                divmu=Dpx(muc(:,:,1),dx);
                divmu=divmu+Dpy(muc(:,:,2),dy);
%                 fdivq=fft2(divq);
%                 fdivmu=fft2(divmu);
%                 nom=((fIc)*options.lambda+stability_factor*fft2(u(:,:,d))-(fdivmu+r*fdivq))-fft2(mu2(:,:,d))+r2*fft2(v(:,:,d));
%                 denom=(options.lambda-r*flap+stability_factor+r2);
%                 Ifn=nom./denom;
                        rhsvec=(options.lambda*(options.fidelity_weight(:).*IRLS_weights(:).*Ic(:))+stability_factor*uc(:)-r*divq(:)-divmu(:)-mu2c(:)+r2*vc(:));
                        sol=u1(:,:,d);sol=sol(:);
%                         ressss=[];
                        for gs_i=1:10
                        sol2=LHS_L\(rhsvec-LHS_U*sol);
%                         ressss(end+1)=norm(sol(:)-sol2(:));
                        sol=sol2;
                        end
                
                u(:,:,d)=reshape(sol,size(u(:,:,d)));
            end
            if (options.use_rre)
                iterants_u(:,end+1)=u(:);
                iterants_mu(:,end+1)=mu(:);
                iterants_q(:,end+1)=q(:);
                if (size(iterants_u,2)>10)
                    s=RRE(iterants_u(:,3:end));
                    u=reshape(s,size(u));
                    s=RRE(iterants_mu(:,3:end));
                    mu=reshape(s,size(mu));
                    s=RRE(iterants_q(:,3:end));
                    q=reshape(s,size(q));
                    iterants_mu=[];
                    iterants_u=[];
                    iterants_q=[];
                end
            end
            
            gu=q;
            % compute the new gradient of u
            for d =1:color_dim
                uc=u(:,:,d);
                gu(:,:,2*(d-1)+1)=Dmx(uc,dx);
                gu(:,:,2*(d-1)+2)=Dmy(uc,dy);
            end
            w=r*gu-mu;
            nw=sqrt(sum(w.^2,3));
            if (isempty(options.diffusivity))
            qn=1/r*repmat(max(0,1-1./(nw+1e-15)),[1 1 size(w,3)]).*w;
            else
                qn=1/r*repmat(max(0,1-options.diffusivity./(nw+1e-15)),[1 1 size(w,3)]).*w;
            end
            resss=[];
            %             [qn,resss]=update_q_belt_al(q,mu,gu,[beta1,beta2,beta3,r,options.iter_q_1,options.iter_q_2]);
            ires_cnt=ires_cnt+1;
            resss=resss(resss>0);
            stats.ressss=[stats.ressss;resss];
            stats.nresss(end+1)=sum(resss>0);
            ires(ires_cnt)=norm(qn(:)-q(:));
            
            q=qn;
%             v=project_onto_se3(u-mu2/r2);
            vold=v;
%             v=project_onto_se3((r2*u-mu2+vold*options.stability_factor)./(r2+options.stability_factor));
            v=project_onto_s2((r2*u+vold*stability_factor+mu2)./(r2+stability_factor));
            res_cnt=res_cnt+1;
            res(res_cnt)=norm(u(:)-u0(:));
            stats.gaps(res_cnt)=norm((q(:)-gu(:))/numel(q(:,:,1)));
            stats.gaps2(res_cnt)=norm((v(:)-u(:))/numel(v(:,:,1)));
            if (res(res_cnt))<1e-12
                break;
            end
%             E_SO3=(u(:,:,1).*u(:,:,1)+u(:,:,4).*u(:,:,4)+u(:,:,7).*u(:,:,7)-1).^2;
%             E_SO3=E_SO3+(u(:,:,1).*u(:,:,2)+u(:,:,4).*u(:,:,5)+u(:,:,7).*u(:,:,8)).^2*2;
%             E_SO3=E_SO3+(u(:,:,1).*u(:,:,3)+u(:,:,4).*u(:,:,6)+u(:,:,7).*u(:,:,9)).^2*2;
%             E_SO3=(u(:,:,2).*u(:,:,2)+u(:,:,5).*u(:,:,5)+u(:,:,8).*u(:,:,8)-1).^2;
%             E_SO3=(u(:,:,3).*u(:,:,3)+u(:,:,6).*u(:,:,6)+u(:,:,9).*u(:,:,9)-1).^2;
%             E_SO3=E_SO3+(u(:,:,2).*u(:,:,3)+u(:,:,5).*u(:,:,6)+u(:,:,8).*u(:,:,9)).^2*2;
%             stats.orth_dev(end+1)=mean(E_SO3(:));
            try
                if (cputime-t1>1) && options.verbose
                    t1=cputime;
                    %                     subplot(231);imshow(u((options.padding+1):(end-options.padding),(options.padding+1):(end-options.padding),:),[]);
                    subplot(231);
%                     unz=u./repmat(sqrt(sum(u.^2,3)),[1 1 size(u,3)]);
%                     vis(:,:,1)=sum(unz.*repmat(permute([1 0 0 0 1 0 0 0 1 0 0 0]',[3 2 1]),[size(u,1),size(u,2),1]),3)/sqrt(3);
%                     vis(:,:,2)=sum(unz.*repmat(permute([0 1 0 -1 0 0 0 0 1 0 1 0]',[3 2 1]),[size(u,1),size(u,2),1]),3)/sqrt(3);
%                     vis(:,:,3)=sum(unz.*repmat(permute([0 0 1 0 1 0 -1 0 0 1 0 0]',[3 2 1]),[size(u,1),size(u,2),1]),3)/sqrt(3);
%                     for iii=1:3
%                         vis(:,:,iii)=vis(:,:,iii)-min(min(vis(:,:,iii)));
%                         vis(:,:,iii)=vis(:,:,iii)/max(median(vis(:,:,iii)));
%                         
%                     end
                    imshow(u/2+0.5,[])
                    title('current solution')
                    if (res_cnt>1)
                        subplot(232);semilogy(res(2:res_cnt));
                        title('u iterations residuals')
                        subplot(233);semilogy(res2(2:res2_cnt));
                        title('u outer iterations residuals')
                        %                 if (exist('ires','var'))
                        %                     subplot(234);semilogy(ires(2:ires_cnt));
                        %                     title('q iterations residuals')
                        %                 end
                    end
%                     subplot(236);semilogy(stats.gaps2);
                    title('duality gap')
                    subplot(235);imshow(I/2+0.5,[])
                    subplot(236);semilogy(stats.orth_dev);
                    title('orthogonality gap')
                    drawnow;
                end
            catch
            end
        end
        for d = 1:color_dim
            qc=q(:,:,(1:2)+(d-1)*2);
            muc=mu(:,:,(1:2)+(d-1)*2);
            uc=u(:,:,d);
            vc=v(:,:,d);
            muc2=mu2(:,:,d);
            gradu=Dmx(uc,dx);
            gradu(:,:,2)=Dmy(uc,dy);
            
            mu_new=muc+r*(qc-gradu);
            mu(:,:,(1:2)+(d-1)*2)=mu_new;
            mu2_new=muc2+r2*(uc-vc);
            mu2(:,:,d)=mu2_new;
        end
        res2_cnt=res2_cnt+1;
        ts(res2_cnt)=cputime-t0;
        res2(res2_cnt)=norm(u(:)-u1(:));
        if ~isempty(options.I_ideal)
            stats.psnr(res2_cnt)=-10*log10(norm(options.I_ideal(:)-u(:))^2/numel(u));
        end
        try
            if (mod(j,3)==0) && (options.verbose) && (toc>1)
                tic
                if (~isempty(options.I_ideal))
                    subplot(234);plot(stats.psnr(1:res2_cnt));
                else
                    subplot(234);imshow(sum((u-u1).^2,3),[]);
                    title('residual image');
                end
                subplot(233);semilogy(res2(2:res2_cnt));
                title('u outer iterations residuals')
                drawnow;
                
                if (~isempty(options.save_video))
                    figure(1)
                    F = getframe(gcf);
                    mov=addframe(mov,F);
                end
                
            end
        catch
        end
    end
    % update r for the augmented Lagrangian penalty
    r=r*2;
    rz=rz*2;
    stability_factor=stability_factor*5;
end
stats.res=res;
stats.ts=ts;
stats.res2=res2;
stats.ires=ires;
stats.v=v;
if (options.compute_metric)
    stats.detg=1./options.beta^4;
    for i = 1:size(u,3)
        qix=Dmx(u(:,:,i),dx);
        stats.detg=stats.detg+1/options.beta^2*qix.^2;
        qiy=Dmy(u(:,:,i),dy);
        stats.detg=stats.detg+1/options.beta^2*qiy.^2;
        for j = 1:size(u,3)
            if (j==i)
                continue;
            end
            qjx=Dmx(u(:,:,j));
            qjy=Dmy(u(:,:,j));
            stats.detg=stats.detg+(qix.*qjy-qiy.*qjx).^2;
        end
    end
    
end
if (options.padding>0)
    u=u((options.padding+1):(end-options.padding),(options.padding+1):(end-options.padding),:);
end
if (~isempty(options.save_video))
    mov=close(mov);
end

end
function res=Dpxc(I,options)
switch (lower(options.scheme))
    case 'backward_forward'
        res=(I(:,[2:end,1])-I);
    case 'backward_forward2'
        res2=imfilter(I,([0 0 -3/2 2 -1/2]),0);
        res=(I(:,[2:end,1])-I);
        res(6:(end-5),6:(end-5),:)=res2(6:(end-5),6:(end-5),:);
    otherwise
        res=(I(:,[2:end,end])-I(:,[1, 1:(end-1)]))/2;
end
end
function res=Dmxc(I,options)
switch (lower(options.scheme))
    case 'backward_forward'
        res=(I-I(:,[end,1:(end-1)]));
    case 'backward_forward2'
        res2=imfilter(I,[1/2 -2 3/2 0 0 ],0);
        res=(I-I(:,[end,1:(end-1)]));
        res(6:(end-5),6:(end-5),:)=res2(6:(end-5),6:(end-5),:);
    otherwise
        res=(I(:,[2:end,end])-I(:,[1,1:(end-1)]))/2;
end
end
function res=Dpyc(I,options)
res=Dpxc(I',options)';
end
function res=Dmyc(I,options)
res=Dmxc(I',options)';
end
function res=create_fft_rep(flt,siz)
zlap=zeros(siz);
zlap(1)=1;
bndcond='circular';
zlap=fftshift(zlap);
K=imfilter(zlap,flt,bndcond);
res=fft2(ifftshift(K));
end