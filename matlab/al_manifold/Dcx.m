function res=Dcx(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=(phi(:,[2:(end),end],:)-phi(:,[1,1:(end-1)],:))/2/h;
end