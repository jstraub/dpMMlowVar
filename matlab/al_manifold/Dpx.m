function res=Dpx(phi,h)
if (exist('h','var')==0)
    h=1;
end
if (numel(size(phi))==3)
res=(phi(:,[2:(end),end],:)-phi)/h;
elseif (numel(size(phi))==4)
    res=(phi(:,[2:(end),end],:,:)-phi)/h;
elseif (numel(size(phi))==2)
    res=(phi(:,[2:(end),end])-phi)/h;
end
end