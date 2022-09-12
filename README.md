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
module use /opt/scorec/spack/dev/lmod/linux-rhel7-x86_64/Core
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core
module load gcc/7.4.0-c5aaloy cuda/10.2 cmake

function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cuda
  echo "build-${name}-${buildSuffix}"
}
export kk=$root/`getname kokkos`/install
export oh=$root/`getname omegah`/install
export cab=$root/`getname cabana`/install
CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$cab$CMAKE_PREFIX_PATH

cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kk"
```


Create a file named `buildAll_turing.sh` with the following contents:

```
#!/bin/bash -e

#kokkos
cd $root
git clone -b 3.4.01 git@github.com:kokkos/kokkos.git
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

## run 

The following assumes that the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies.

```
./build-gyroScatterEff-cranium-cuda/gyroScatterEff run/gyroScatterData0
```

If all goes well the following output should appear:

```
version 0.1.0
done
```

## timing data

... coming soon ...

we'll use kokkos profiling 
https://github.com/kokkos/kokkos-tools/wiki/SimpleKernelTimer
and some of the others listed here:
https://github.com/kokkos/kokkos-tools/wiki

