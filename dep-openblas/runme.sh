#!/usr/bin/bash

# This script is a build recipe to clone and build OpenBLAS from source correctly and easily on Banana PI F3.
base_dir=$(pwd)

# select gcc14.2 with update-alternative, this is because gcc-13 does not support VLA/VLS and some RVV stuff.
sudo update-alternatives --set gcc /usr/bin/gcc-14 || exit 1
sudo update-alternatives --set g++ /usr/bin/g++-14 || exit 1

# clone OpenBLAS from github
wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz || exit 2
tar -xvf OpenBLAS-0.3.29.tar.gz || exit 3
cd OpenBLAS-0.3.29 || exit 4
ob_root=$(pwd)
mkdir build && mkdir install_dir && cd build || exit 5
ob_install_dir=${ob_root}/install_dir
ob_build_dir=${ob_root}/build

cmake -DARCH=riscv64 \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_LAPACK_DEPRECATED=ON \
      -DBUILD_RELAPACK=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DBUILD_TESTING=ON \
      -DBUILD_WITHOUT_CBLAS=OFF \
      -DBUILD_WITHOUT_LAPACK=OFF \
      -DBUILD_WITHOUT_LAPACKE=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${ob_install_dir} \
      -DCPP_THREAD_SAFETY_GEMV=OFF \
      -DCPP_THREAD_SAFETY_TEST=OFF \
      -DC_LAPACK=ON \
      -DDYNAMIC_ARCH=OFF \
      -DDYNAMIC_OLDER=OFF \
      -DFIXED_LIBNAME=OFF \
      -DNO_AFFINITY=ON \
      -DNO_WARMUP=ON \
      -DUSE_LOCKING=ON \
      -DUSE_OPENMP=ON \
      -DUSE_PERL=OFF .. || exit 5

make -j 4 || exit 6  # use just -j 4 to limit the memory usage. We only have 4G of RAM on the board.
make install || exit 7

cat << 'EOF' > "${base_dir}/env.sh"
#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed."
    exit 1
fi

# check to see if the environment variable MyOpenBLAS is defined or not; if its defined, skip exporting paths, if not defined, export the paths
if [[ -z "${MyOpenBLAS}" ]]; then
    export MyOpenBLAS=1
    export LD_LIBRARY_PATH=${ob_install_dir}/lib:$LD_LIBRARY_PATH
    export CMAKE_PREFIX_PATH=${ob_install_dir}:$CMAKE_PREFIX_PATH
fi
EOF

echo "OpenBLAS build and installation completed successfully."
echo "Please source the env.sh file in ${base_dir}/env.sh to set up the environment variables."
echo "You can source it in your bashrc. It is smart enough not to export the paths if they are already exported."