Mex wrapper for dpMMlowVar library; currently only supports CPU-based, Euclidean
DP-Means and DDP-Means, but would be straightforward to include flags for
spherical, gpu, etc.

To Install:
1. Build libdpMMlowVar library using CMake, as normal. It may be easier to
   comment out all the tests and and sample code, so that the library is all
   that gets built.
2. Open Matlab in the matlab/mex directory and type 'make'
   For this to work, you must 
     a) have all custom include directories (if any) in CPPFLAGS
     b) have libdpMMlowVar.so located at build/lib (do this by building CMake
        normally, ensuring that the directory you build in is called build)

To Run:
1. Assuming the mex commands compiled successfully, you should be able to run
   the demos: ddp_demo and dpmm_demo.
