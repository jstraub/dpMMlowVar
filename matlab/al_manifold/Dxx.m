function res=Dxx(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=(phi(:,[2:(end),end])-2*phi+phi(:,[1,1:(end-1)]))/h^2;
end