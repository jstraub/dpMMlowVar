function res=Dyy(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=permute(Dxx(permute(phi,[2 1]),h),[2 1]);
end