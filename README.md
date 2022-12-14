# gyroScatterEff
isolated gyroScatterEff kernel

## build dependencies

The following commands were tested on a SCOREC workstation running RHEL7 with a
Nvidia Turing GPU.

`cd` to a working directory that will contain *all* your source code (including
this directory) and build directories.  That directory is referred to as `root`
in the following bash scripts.

Create a file named `envRhel7_turing.sh` with the following contents:

```
export root=$PWD 
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core 
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core 
module load gcc/10.1.0 cmake cuda/11.4

function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cuda
  echo "build-${name}-${buildSuffix}"
}
export kk=$root/`getname kokkos`/install
export oh=$root/`getname omegah`/install
export cab=$root/`getname cabana`/install
export mf=$root/`getname meshFields`/install
CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$cab:$mf:$CMAKE_PREFIX_PATH

cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kk"
```


Create a file named `buildAll_turing.sh` with the following contents:

```
#!/bin/bash -e

#kokkos
cd $root
#tested with kokkos develop@9dff8cc
git clone -b develop git@github.com:kokkos/kokkos.git
mkdir -p $kk
cd $_/..
cmake ../kokkos \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DKokkos_ENABLE_PROFILING=on \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24 install

#omegah
cd $root
git clone git@github.com:sandialabs/omega_h.git
[ -d $oh ] && rm -rf ${oh%%install}
mkdir -p $oh 
cd ${oh%%install}
cmake ../omega_h \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=OFF  \
  -DBUILD_TESTING=on  \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_PREFIX=$kk/lib64/cmake
make VERBOSE=1 -j8 install
ctest

#cabana
cd $root
git clone git@github.com:ECP-copa/Cabana.git cabana
[ -d $cab ] && rm -rf ${cab%%install}
mkdir -p $cab
cd ${cab%%install}
cmake ../cabana \
  -DCMAKE_BUILD_TYPE="Debug" \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DCabana_ENABLE_MPI=OFF \
  -DCabana_ENABLE_CAJITA=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCabana_ENABLE_Cuda=ON \
  -DCMAKE_INSTALL_PREFIX=$cab
make -j 24 install

#meshFields
cd $root
git clone git@github.com:SCOREC/meshFields.git
cmake -S meshFields -B ${mf%%install} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=$mf
cmake --build ${mf%%install} -j2 --target install
```

Make the script executable:

```
chmod +x buildAll_turing.sh
```


Source the environment script from this work directory:

```
source envRhel7_turing.sh
```

Run the build script:

```
./buildAll_turing.sh
```

## download kernel input data

The following assumes that the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies.

Note, this is ~500MB of data.  

```
mkdir $root/run 
cd $_
wget https://zenodo.org/record/7072575/files/gyroScatterData0.txt
wget https://zenodo.org/record/7072575/files/gyroScatterData0_bmap.bin
wget https://zenodo.org/record/7072575/files/gyroScatterData0_fmap.bin
wget https://zenodo.org/record/7072575/files/gyroScatterData0_owners.bin
```


## build gyroScatterEff

The following assumes that the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies.

```
cd $root
git clone git@github.com:SCOREC/gyroScatterEff
cmake -S gyroScatterEff -B build-gyroScatterEff-cuda -DDATA_DIR=run/
cmake --build build-gyroScatterEff-cuda 
```

## rebuild gyroScatterEff

The following assumes that (1) the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies and (2) that `gyroScatterEff` was previously built.

Use the following command to rebuild `gyroScatterEff` after making changes to its source code.  This command should be run from the `$root` directory.

```
cmake --build build-gyroScatterEff-cuda 
```

## run 

The following assumes that the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies.

```
./build-gyroScatterEff-cuda/gyroScatterEff run/gyroScatterData0 0 10
```

Where `0` specifies use of the Omega_h arrays, and `10` is the number of times to run the kernel.

Specifying `1` for the first argument will use the Cabana 'packed' AoSoA and `2` will use the Cabana 'split' AoSoAs (still a work in progress).

If all goes well the following output should appear:

```
version 0.2.0
done
```

## Timing Data

For timing, we'll use Kokkos SimpleKernelTimer along with some of the other tools listed here:
https://github.com/kokkos/kokkos-tools/wiki

First, clone the repository
```
git clone git@github.com:kokkos/kokkos-tools.git
```
Then, follow the instructions here to compile:
https://github.com/kokkos/kokkos-tools/wiki/SimpleKernelTimer

To run your program with timing data, run the following from your `root` directory
```
export KOKKOS_PROFILE_LIBRARY={PATH_TO_TOOL_DIRECTORY}/kp_kernel_timer.so
./build-gyroScatterEff-cuda/gyroScatterEff run/gyroScatterData0 0 10
```
This will produce normal program output, plus the following line
```
KokkosP: Kernel timing written to {timing output file path}
```

To view the timing data, run the following command
```
{PATH_TO_TOOL_DIRECTORY}/kp_reader {timing output file path}
```
