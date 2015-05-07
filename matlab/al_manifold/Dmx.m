function res=Dmx(phi,h)
if (exist('h','var')==0)
    h=1;
end
res=(phi-phi(:,[1,1:(end-1)],:,:))/h;
end