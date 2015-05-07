function res=Dcw(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=permute(Dcx(permute(phi,[1 4 3 2]),h),[1 4 3 2]);
end