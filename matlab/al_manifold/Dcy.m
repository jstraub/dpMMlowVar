function res=Dcy(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=permute(Dcx(permute(phi,[2 1 3]),h),[2 1 3]);
end