function v=project_onto_s2(u)
d=numel(size(u));
n=sqrt(sum(u.^2,d)+eps);
v=bsxfun(@rdivide,u,n);
end