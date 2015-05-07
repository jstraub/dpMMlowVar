function res=Dmz(phi,h)
if (exist('h','var')==0)
    h=1;
end
if (ndims(phi)==3)
res=permute(Dmx(permute(phi,[1 3 2]),h),[1 3 2]);
elseif (ndims(phi)==4)
res=permute(Dmx(permute(phi,[1 3 2 4]),h),[1 3 2 4]);
else
end
end