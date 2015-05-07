function res=Dcz(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=permute(Dcx(permute(phi,[1 3 2]),h),[1 3 2]);
end