function res=Dpz(phi,h)
if (exist('h','var')==0)
    h=1;
end
if (numel(size(phi))==3)
res=permute(Dpx(permute(phi,[1 3 2]),h),[1 3 2]);
elseif (numel(size(phi))==4)
    res=permute(Dpx(permute(phi,[1 3 2 4]),h),[1 3 2 4]);
end
end