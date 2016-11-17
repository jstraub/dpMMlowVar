% Must remove shared objects (libdpMMlowVar.so) from memory if its been used.
clear mex_dpmm
clear mex_ddp

% Copy already-compiled dpMMlowVar shared library from what we presume is build
system('cp ../../build/lib/libdpMMlowVar.so .');

% Get include flags from CPPFLAGS as matlab's mex doesn't seem to honor them.
[~, flags] = system('echo $CPPFLAGS');
cppIncludes = regexp(flags, '-I[^\s]*', 'match');
cppIncludes{end+1} = '-I../../include';
cppIncludes = cellfun(@(c)[c(1:2) '"' c(3:end) '"'],cppIncludes, ...
  'UniformOutput',0);
includes = strjoin(cppIncludes);

try
  fprintf('Building mex_dpmm.cpp...\n');
  mex(includes, '-ldpMMlowVar', 'mex_dpmm.cpp');
  fprintf('Building mex_ddp.cpp...\n');
  mex(includes, '-ldpMMlowVar', 'mex_ddp.cpp');
  fprintf('Build successful; you should be able to run ddp_demo & dpmm_demo.\n');
catch
  fprintf('Build not successful; try running mex with -v flag\n');
end
