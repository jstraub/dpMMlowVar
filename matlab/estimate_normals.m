function n=estimate_normals(x,y,z,W)
flt=ones(2*W+1);
% xx=imfilter(x.*x,flt,'replicate');
% xy=imfilter(x.*y,flt,'replicate');
% xz=imfilter(x.*z,flt,'replicate');
% yy=imfilter(y.*y,flt,'replicate');
% yz=imfilter(y.*z,flt,'replicate');
% zz=imfilter(z.*z,flt,'replicate');
% xi=imfilter(x,flt,'replicate');
% yi=imfilter(y,flt,'replicate');
% zi=imfilter(z,flt,'replicate');
% 
% a= (xi.*yz.^2 - xz.*yi.*yz - xi.*yy.*zz + xy.*yi.*zz - xy.*yz.*zi + xz.*yy.*zi)./(zz.*xy.^2 - 2*xy.*xz.*yz + yy.*xz.^2 + xx.*yz.^2 - xx.*yy.*zz);
% b= (xz.^2.*yi - xi.*xz.*yz + xi.*xy.*zz - xy.*xz.*zi - xx.*yi.*zz + xx.*yz.*zi)./(zz.*xy.^2 - 2*xy.*xz.*yz + yy.*xz.^2 + xx.*yz.^2 - xx.*yy.*zz);
% c= (xy.^2.*zi - xi.*xy.*yz + xi.*xz.*yy - xy.*xz.*yi + xx.*yi.*yz - xx.*yy.*zi)./(zz.*xy.^2 - 2*xy.*xz.*yz + yy.*xz.^2 + xx.*yz.^2 - xx.*yy.*zz);
% n=sqrt(a.^2+ b.^2+c.^2);
% a=a./n;b=b./n;c=c./n;
flt_x=[1 0 -1];flt_x=[flt_x;flt_x;flt_x];
xu=imfilter(x,flt_x,'replicate');
xv=imfilter(x,flt_x','replicate');
yu=imfilter(y,flt_x,'replicate');
yv=imfilter(y,flt_x','replicate');
zu=imfilter(z,flt_x,'replicate');
zv=imfilter(z,flt_x','replicate');

xu1=imfilter(xu,flt,'replicate');
xv1=imfilter(xv,flt,'replicate');
yu1=imfilter(yu,flt,'replicate');
yv1=imfilter(yv,flt,'replicate');
zu1=imfilter(zu,flt,'replicate');
zv1=imfilter(zv,flt,'replicate');

nu=sqrt(xu1.^2+yu1.^2+zu1.^2);
nv=sqrt(xv1.^2+yv1.^2+zv1.^2);
xu1=xu1./nu;
yu1=yu1./nu;
zu1=zu1./nu;
xv1=xv1./nv;
yv1=yv1./nv;
zv1=zv1./nv;
n=[];
n.nx=yu1.*zv1 - yv1.*zu1;
n.ny=xv1.*zu1 - xu1.*zv1;
n.nz=xu1.*yv1 - xv1.*yu1;
nn=sqrt(n.nx.^2+n.ny.^2+n.nz.^2+eps);
sg=sign(x.*n.nx+y.*n.ny+z.*n.nz)./nn;
n.nx=n.nx.*sg;
n.ny=n.ny.*sg;
n.nz=n.nz.*sg;
nn=zeros(size(x,1),size(x,2),3);
nn(:,:,1)=n.nx;
nn(:,:,2)=n.ny;
nn(:,:,3)=n.nz;
%figure()
%imshow(nn)
%figure()
%plot3(nn(:,:,1),nn(:,:,2),nn(:,:,3),'.')
end
