function res=Dmy(phi,h)
if (exist('h','var')==0)
    h=1;
end
if (ndims(phi)==2)
res=permute(Dmx(permute(phi,[2 1]),h),[2 1]);
elseif (ndims(phi)==3)
res=permute(Dmx(permute(phi,[2 1 3]),h),[2 1 3]);
elseif (ndims(phi)==4)
res=permute(Dmx(permute(phi,[2 1 3 4]),h),[2 1 3 4]);
else
end
end