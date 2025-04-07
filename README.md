# HPC-Course-Projects
The project follows CBLAS interface losely (some arguments are changed to be C++ friendly). 
The main structure is implemented using templates to avoid polymorphism and its runtime overhead. 
You can find runtime measurement, tensor, and some other utility classes in `common/`.

## Structure

- benchmark: Any benchmarking target for one or more BLAS functions should be implemented here. Please follow the example.
- CMakeLists.txt: The main CMake script of the project.
- common: The directory containing the sources for the `common` static library.
- dep-openblas: The utility scripts to clone and build OpenBLAS correctly on a RV64-RVV-1.0 machine. 
- src: The directory containing the sources for the `blas` static library.
- test: The directory containing all the unit tests (NIY).


Please feel free to contact me on Discord. You can use the repo's discussion page as well.

