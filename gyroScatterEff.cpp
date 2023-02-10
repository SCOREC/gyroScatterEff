#include <Omega_h_library.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <cstdlib> //exit_failure
#include <fstream> //std::ifstream
#include "gyroScatterData0.txt" //defines problem size constants
#include <Cabana_Core.hpp>
#include <MeshField.hpp>

// TODO:
#include <cmath> //std::floor


namespace oh = Omega_h;
namespace cab = Cabana;

namespace {
  const int VectorLength = 32;
  using MemorySpace = Kokkos::CudaSpace;
  using ExecutionSpace = Kokkos::Cuda;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  template <typename T>
  oh::Read<T> readArrayBinary(std::string fname) {
    std::ifstream inBin(fname, std::ios::binary);
    if(!inBin.is_open()) {
      fprintf(stderr, "problem reading %s\n", fname.c_str());
      exit(EXIT_FAILURE);
    }
    const auto compressed = false;
    const auto needs_swapping = !oh::is_little_endian_cpu();
    oh::Read<T> array;
    oh::binary::read_array(inBin, array, compressed, needs_swapping);
    inBin.close();
    return array;
  }

  //the following functions are from SCOREC/xgcm @ 8774ee9 src/xgcm_scatter_map.hpp
  OMEGA_H_DEVICE oh::LO mappedVertex(oh::LOs point_map, const oh::LO vtx, const oh::LO ring,
      const oh::LO point, const oh::LO vtx_of_elm) {
    return point_map[numVertsPerElm * (numPtsPerRing * (numRings * vtx + ring) + point) + vtx_of_elm];
  }

  OMEGA_H_DEVICE oh::Real mappedWeight(oh::Reals point_weights, const oh::LO vtx, const oh::LO ring,
      const oh::LO point, const oh::LO vtx_of_elm) {
    return point_weights[numVertsPerElm * (numPtsPerRing * (numRings * vtx + ring) + point) + vtx_of_elm];
  }
};

void gyroScatterOmegah(oh::Reals e_half,
    oh::LOs& forward_map, oh::LOs& backward_map,
    oh::Reals& forward_weights, oh::Reals& backward_weights,
    oh::Write<oh::Real> eff_major, oh::Write<oh::Real> eff_minor,
    const oh::LO gnrp1, const oh::LO gppr,
    oh::LOs& owners) {
  const int ncomps = e_half.size() / (2 * numVerts);
  assert(ncomps == numComponents);
  int mesh_rank = 0;
  const oh::LO nvpe = numVertsPerElm;
  // handle ring = 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_ring0_region");
  auto efield_scatter_ring0 = OMEGA_H_LAMBDA(const int vtx) {
    // index on gyro averaged electric field on ring=0
    const auto index = vtx * gnrp1 * ncomps;
    const oh::LO gyroVtxIdx_f = 2 * (vtx * ncomps) + 1;
    const oh::LO gyroVtxIdx_b = 2 * (vtx * ncomps);
    for (int i = 0; i < ncomps; ++i) {
      const oh::LO ent = index + i * gnrp1;
      assert(ent<effMinorSize);
      assert((gyroVtxIdx_f + 2 * i) < ehalfSize);
      assert((gyroVtxIdx_b + 2 * i) < ehalfSize);
      eff_major[ent] = e_half[gyroVtxIdx_f + 2 * i];
      eff_minor[ent] = e_half[gyroVtxIdx_b + 2 * i];
    }
  };
  oh::parallel_for(numVerts, efield_scatter_ring0, "efield_scatter_ring0");
  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());

  // handle ring > 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_region");
  auto efield_scatter = OMEGA_H_LAMBDA(const int vtx) {
    if (owners[vtx] == mesh_rank) {
      for(int ring=1; ring < gnrp1; ring++) {
        // index on gyro averaged electric field
        const auto index = vtx * gnrp1 * ncomps + ring;
        for(int pt=0; pt<gppr; pt++) {
          for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
            const auto mappedVtx_f = mappedVertex(forward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_f = mappedWeight(forward_weights, vtx, ring, pt, elmVtx);
            const auto mappedVtx_b = mappedVertex(backward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_b = mappedWeight(backward_weights, vtx, ring, pt, elmVtx);

            // Only compute contributions of owned vertices.
            // Field Sync will sum all contributions.
            // This part of the operation is basically a Matrix (sparse matrix)
            // and vector multiplication: c_j = A_ij * b_j, where vector b and
            // c are vectors defined on the mesh vertices, with b_j, c_j the
            // value at vertex j; while A is the gyro-average mapping matrix,
            // A_ij represents the mapping weight from vertex i to vertex j
            // (from field vector b to field vector c). We need to make sure
            // the index is correct in performing this operation
            if (mappedVtx_f >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the major component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_f = 2 * (mappedVtx_f * ncomps + i) + 1;
                Kokkos::atomic_add(&(eff_major[index + i * gnrp1]),
                    mappedWgt_f * e_half[gyroVtxIdx_f] / gppr);
              }
            }
            if (mappedVtx_b >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the minor component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_b = 2 * (mappedVtx_b * ncomps + i);
                // TODO:atomic increment -> sanity check (compare to kokkos version)
                Kokkos::atomic_add(&(eff_minor[index + i * gnrp1]),
                    mappedWgt_b * e_half[gyroVtxIdx_b] / gppr);
              }
            }
          }
        }
      }
    }
  };
  oh::parallel_for(numVerts, efield_scatter, "gyroScatterEFF");
  Kokkos::Profiling::popRegion();
}

template<class EffSlice>
void gyroScatterCab(oh::Reals e_half,
    oh::LOs& forward_map, oh::LOs& backward_map,
    oh::Reals& forward_weights, oh::Reals& backward_weights,
    EffSlice& eff_major, EffSlice& eff_minor,
    const oh::LO gnrp1, const oh::LO gppr,
    oh::LOs& owners, std::string modeName) {
  const int ncomps = e_half.size() / (2 * numVerts);
  assert(ncomps == numComponents);
  int mesh_rank = 0;
  const oh::LO nvpe = numVertsPerElm;
  // handle ring = 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_ring0_cab"+modeName+"_region");
  auto efield_scatter_ring0_cab = KOKKOS_LAMBDA(const int s, const int a) {
    const auto vtx = s*VectorLength+a;
    // index on gyro averaged electric field on ring=0
    const oh::LO gyroVtxIdx_f = 2 * (vtx * ncomps) + 1;
    const oh::LO gyroVtxIdx_b = 2 * (vtx * ncomps);
    for (int i = 0; i < ncomps; ++i) {
      assert((gyroVtxIdx_f + 2 * i) < ehalfSize);
      assert((gyroVtxIdx_b + 2 * i) < ehalfSize);
      const oh::LO ent = i * gnrp1;
      eff_major.access(s, a, ent) = e_half[gyroVtxIdx_f + 2 * i];
      eff_minor.access(s, a, ent) = e_half[gyroVtxIdx_b + 2 * i];
    }
  };
  cab::SimdPolicy<VectorLength, ExecutionSpace> simd_policy(0, numVerts);
  cab::simd_parallel_for(simd_policy, efield_scatter_ring0_cab, "efield_scatter_ring0_cab"+modeName);
  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());

  // handle ring > 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_cab"+modeName+"_region");
  auto efield_scatter_cab = KOKKOS_LAMBDA(const int s, const int a) {
    const auto vtx = s*VectorLength+a;
    if (owners[vtx] == mesh_rank) {
      for(int ring=1; ring < gnrp1; ring++) {
        // index on gyro averaged electric field
        for(int pt=0; pt<gppr; pt++) {
          for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
            const auto mappedVtx_f = mappedVertex(forward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_f = mappedWeight(forward_weights, vtx, ring, pt, elmVtx);
            const auto mappedVtx_b = mappedVertex(backward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_b = mappedWeight(backward_weights, vtx, ring, pt, elmVtx);

            // Only compute contributions of owned vertices.
            // Field Sync will sum all contributions.
            // This part of the operation is basically a Matrix (sparse matrix)
            // and vector multiplication: c_j = A_ij * b_j, where vector b and
            // c are vectors defined on the mesh vertices, with b_j, c_j the
            // value at vertex j; while A is the gyro-average mapping matrix,
            // A_ij represents the mapping weight from vertex i to vertex j
            // (from field vector b to field vector c). We need to make sure
            // the index is correct in performing this operation
            if (mappedVtx_f >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the major component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_f = 2 * (mappedVtx_f * ncomps + i) + 1;
                Kokkos::atomic_add(&eff_major.access(s, a, i * gnrp1 + ring),
                    mappedWgt_f * e_half[gyroVtxIdx_f] / gppr);
              }
            }
            if (mappedVtx_b >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the minor component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_b = 2 * (mappedVtx_b * ncomps + i);
                Kokkos::atomic_add(&eff_minor.access(s, a, i * gnrp1 + ring),
                    mappedWgt_b * e_half[gyroVtxIdx_b] / gppr);
              }
            }
          }
        }
      }
    }
  };
  cab::simd_parallel_for(simd_policy, efield_scatter_cab, "gyroScatterEFF_cab"+modeName);
  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());
}


template<class MeshField, class MeshFieldController>
void gyroScatterMeshFields(oh::Reals e_half,
    oh::LOs& forward_map, oh::LOs& backward_map,
    oh::Reals& forward_weights, oh::Reals& backward_weights,
    MeshField& eff_major, MeshField& eff_minor,
    MeshFieldController& mfCon,
    const oh::LO gnrp1, const oh::LO gppr,
    oh::LOs& owners, std::string modeName) {
  const int ncomps = e_half.size() / (2 * numVerts);
  assert(ncomps == numComponents);
  int mesh_rank = 0;
  const oh::LO nvpe = numVertsPerElm;
  // handle ring = 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_ring0_meshField"+modeName+"_region");
  auto efield_scatter_ring0_meshField = KOKKOS_LAMBDA(const int s, const int a) {
    const auto vtx = s*VectorLength+a;
    // index on gyro averaged electric field on ring=0
    const oh::LO gyroVtxIdx_f = 2 * (vtx * ncomps) + 1;
    const oh::LO gyroVtxIdx_b = 2 * (vtx * ncomps);
    for (int i = 0; i < ncomps; ++i) {
      assert((gyroVtxIdx_f + 2 * i) < ehalfSize);
      assert((gyroVtxIdx_b + 2 * i) < ehalfSize);
      const oh::LO ent = i * gnrp1;
      eff_major(s, a, ent) = e_half[gyroVtxIdx_f + 2 * i];
      eff_minor(s, a, ent) = e_half[gyroVtxIdx_b + 2 * i];
    }
  };
  
  mfCon.parallel_for(0, numVerts, efield_scatter_ring0_meshField, "efield_scatter_ring0_meshField"+modeName);
  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());

  // handle ring > 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_meshField"+modeName+"_region");
  auto efield_scatter_meshField = KOKKOS_LAMBDA(const int s, const int a) {
    const auto vtx = s*VectorLength+a;
    if (owners[vtx] == mesh_rank) {
      for(int ring=1; ring < gnrp1; ring++) {
        // index on gyro averaged electric field
        for(int pt=0; pt<gppr; pt++) {
          for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
            const auto mappedVtx_f = mappedVertex(forward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_f = mappedWeight(forward_weights, vtx, ring, pt, elmVtx);
            const auto mappedVtx_b = mappedVertex(backward_map, vtx, ring, pt, elmVtx);
            const auto mappedWgt_b = mappedWeight(backward_weights, vtx, ring, pt, elmVtx);

            // Only compute contributions of owned vertices.
            // Field Sync will sum all contributions.
            // This part of the operation is basically a Matrix (sparse matrix)
            // and vector multiplication: c_j = A_ij * b_j, where vector b and
            // c are vectors defined on the mesh vertices, with b_j, c_j the
            // value at vertex j; while A is the gyro-average mapping matrix,
            // A_ij represents the mapping weight from vertex i to vertex j
            // (from field vector b to field vector c). We need to make sure
            // the index is correct in performing this operation
            if (mappedVtx_f >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the major component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_f = 2 * (mappedVtx_f * ncomps + i) + 1;
                Kokkos::atomic_add(&eff_major(s, a, i * gnrp1 + ring),
                    mappedWgt_f * e_half[gyroVtxIdx_f] / gppr);
              }
            }
            if (mappedVtx_b >= 0) {
              for (int i = 0; i < ncomps; ++i) {
                // access the minor component of e_half
                //TODO: atomic_add probably is not needed here
                const oh::LO gyroVtxIdx_b = 2 * (mappedVtx_b * ncomps + i);
                Kokkos::atomic_add(&eff_minor(s, a, i * gnrp1 + ring),
                    mappedWgt_b * e_half[gyroVtxIdx_b] / gppr);
              }
            }
          }
        }
      }
    }
  };
  mfCon.parallel_for(0, numVerts, efield_scatter_meshField, "gyroScatterEFF_meshField"+modeName);
  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());
}
/*
gyroScatterKokkos( e_half, fmap_d, bmap_d,
                            fweights_d, bweights_d,
                            eff_major, eff_minor,
                            numRings, numPtsPerRing,
                            owners_d, numMajorSOA, numMinorSOA );

*/

void gyroScatterKokkos( oh::Reals e_half, oh::LOs& forward_map,
                        oh::LOs& backward_map, oh::Reals& forward_weights,
                        oh::Reals& backward_weights, Kokkos::View<double*, MemorySpace>& eff_major,
                        Kokkos::View<double*, MemorySpace>& eff_minor, const oh::LO gnrp1, const oh::LO gppr,
                        oh::LOs& owners, int numMajorSOA, int numMinorSOA ) 
{
  typedef typename Kokkos::TeamPolicy<>::member_type member_type;
  
  const int ncomps = e_half.size() / (2 * numVerts);
  assert(ncomps == numComponents);
  int mesh_rank = 0;
  const oh::LO nvpe = numVertsPerElm;
  // handle ring = 0
  
  const int Stride = (effMajorSize/numVerts)*VectorLength;
  int numTuplesInLastSOA = numVerts - (VectorLength*(numMajorSOA-1));
  Kokkos::Profiling::pushRegion("gyroScatterEFF_ring0_kokkos_region");

  auto efield_scatter_ring0_kokkos = KOKKOS_LAMBDA( const member_type& thread ) {
    const int s = thread.league_rank();
    bool isLastSOA = ( numMajorSOA-1 == s );
    int teamSize = isLastSOA ? numTuplesInLastSOA : VectorLength;
    Kokkos::parallel_for(Kokkos::TeamThreadRange( thread, teamSize ), 
    [&] (const int& a) {
      const auto vtx = s*VectorLength+a;
      const oh::LO gyroVtxIdx_f = 2*(vtx*ncomps)+1;
      const oh::LO gyroVtxIdx_b = 2*(vtx*ncomps);
      for (int i = 0; i < ncomps; ++i) {
          assert((gyroVtxIdx_f + 2 * i) < ehalfSize);
          assert((gyroVtxIdx_b + 2 * i) < ehalfSize);
          const oh::LO ent = Stride * s + a + VectorLength*(i*gnrp1);
          eff_major(ent) = e_half[gyroVtxIdx_f + 2 * i];
          eff_minor(ent) = e_half[gyroVtxIdx_b + 2 * i];
      }
    });
  };
   
  Kokkos::parallel_for("gyroScatterEFF_ring0_KokkosView", Kokkos::TeamPolicy<>(numMajorSOA, VectorLength), efield_scatter_ring0_kokkos );

  Kokkos::Profiling::popRegion();
  assert(cudaSuccess==cudaDeviceSynchronize());

  // handle ring > 0
  Kokkos::Profiling::pushRegion("gyroScatterEFF_region");


  auto efield_scatter_kokkos = KOKKOS_LAMBDA( const member_type& thread ) {
    const int s = thread.league_rank();
    bool isLastSOA = ( numMajorSOA-1 == s );
    int teamSize = isLastSOA ? numTuplesInLastSOA : VectorLength;
    
    Kokkos::parallel_for( Kokkos::TeamThreadRange( thread, teamSize ), 
    [&] ( const int& a ) {
        const int vtx = s*VectorLength+a; 
        if (owners[vtx] == mesh_rank) {
          for(int ring=1; ring < gnrp1; ring++) {
            // index on gyro averaged electric field
            for(int pt=0; pt<gppr; pt++) {
              for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
                const auto mappedVtx_f = mappedVertex(forward_map, vtx, ring, pt, elmVtx);
                const auto mappedWgt_f = mappedWeight(forward_weights, vtx, ring, pt, elmVtx);
                const auto mappedVtx_b = mappedVertex(backward_map, vtx, ring, pt, elmVtx);
                const auto mappedWgt_b = mappedWeight(backward_weights, vtx, ring, pt, elmVtx);

                // Only compute contributions of owned vertices.
                // Field Sync will sum all contributions.
                // This part of the operation is basically a Matrix (sparse matrix)
                // and vector multiplication: c_j = A_ij * b_j, where vector b and
                // c are vectors defined on the mesh vertices, with b_j, c_j the
                // value at vertex j; while A is the gyro-average mapping matrix,
                // A_ij represents the mapping weight from vertex i to vertex j
                // (from field vector b to field vector c). We need to make sure
                // the index is correct in performing this operation
                if (mappedVtx_f >= 0) {
                  for (int i = 0; i < ncomps; ++i) {
                    // access the major component of e_half
                    //TODO: atomic_add probably is not needed here
                    const oh::LO gyroVtxIdx_f = 2 * (mappedVtx_f * ncomps + i) + 1;
                    Kokkos::atomic_add(&eff_major(Stride * s + a + VectorLength * (i * gnrp1 + ring)),
                        mappedWgt_f * e_half[gyroVtxIdx_f] / gppr);
                  }
                }
                if (mappedVtx_b >= 0) {
                  for (int i = 0; i < ncomps; ++i) {
                    // access the minor component of e_half
                    //TODO: atomic_add probably is not needed here
                    const oh::LO gyroVtxIdx_b = 2 * (mappedVtx_b * ncomps + i);
                    Kokkos::atomic_add(&eff_minor(Stride * s + a + VectorLength * (i * gnrp1 + ring)),
                        mappedWgt_b * e_half[gyroVtxIdx_b] / gppr);
                  }
                }
              }
            }
          }
        }

    });
  };
  
  Kokkos::parallel_for("gyroScatterEFF_KokkosView",Kokkos::TeamPolicy<>(numMajorSOA, VectorLength), efield_scatter_kokkos );
  Kokkos::Profiling::popRegion();
}


struct version {
  int major;
  int minor;
  int patch;
  void print() const {
    std::string s=std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    std::cout << "version " << s << "\n";
  }
};

int main(int argc, char** argv) {
  const version v{0,3,0};
  v.print();
  if(argc != 4) {
    fprintf(stderr, "Usage: %s <field prefix> <runMode=[0:omegah|1:cabanaPacked|2:cabanaSplit|3:meshFields|4:kokkosView] <iterations>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  std::string fname(argv[1]);
  const auto runMode = atoi(argv[2]);
  const auto numIter = atoi(argv[3]);
  auto lib = Omega_h::Library(&argc, &argv);

  auto fmap_d = readArrayBinary<oh::LO>(fname+"_fmap.bin");
  auto bmap_d = readArrayBinary<oh::LO>(fname+"_bmap.bin");
  auto owners_d = readArrayBinary<oh::LO>(fname+"_owners.bin");

  oh::Reals e_half(ehalfSize);
  oh::Reals fweights_d(forwardWeightsSize);
  oh::Reals bweights_d(backwardWeightsSize);

  if(runMode==0) { //omegah
    fprintf(stderr, "mode: omegah\n");
    oh::Write<oh::Real> eff_major(effMajorSize);
    oh::Write<oh::Real> eff_minor(effMinorSize);

    for(int i=0; i<numIter; i++) {
      gyroScatterOmegah(e_half, fmap_d, bmap_d,
          fweights_d, bweights_d,
          eff_major, eff_minor,
          numRings, numPtsPerRing,
          owners_d);
    }
  } else if(runMode==1) { //packed
    fprintf(stderr, "mode: cabPacked\n");
    constexpr int extent = effMajorSize/numVerts;
    using DataTypes = cab::MemberTypes<double[extent],double[extent]>;
    cab::AoSoA<DataTypes, DeviceType, VectorLength> aosoa("packed", numVerts);
    auto eff_major = cab::slice<0>(aosoa);
    auto eff_minor = cab::slice<1>(aosoa);
    

    for(int i=0; i<numIter; i++) {
      gyroScatterCab(e_half, fmap_d, bmap_d,
          fweights_d, bweights_d,
          eff_major, eff_minor,
          numRings, numPtsPerRing,
          owners_d, std::string("Packed"));
    }
  } else if(runMode==2) { //split
    fprintf(stderr, "mode: cabSplit\n");
    constexpr int extent = effMajorSize/numVerts;
    using DataTypes = cab::MemberTypes<double[extent]>;
    cab::AoSoA<DataTypes, DeviceType, VectorLength> aosoa0("split0", numVerts);
    cab::AoSoA<DataTypes, DeviceType, VectorLength> aosoa1("split1", numVerts);
    auto eff_major = cab::slice<0>(aosoa0);
    auto eff_minor = cab::slice<0>(aosoa1);
    
    for(int i=0; i<numIter; i++) {
      gyroScatterCab(e_half, fmap_d, bmap_d,
		     fweights_d, bweights_d,
		     eff_major, eff_minor,
		     numRings, numPtsPerRing,
		     owners_d, std::string("Split"));
    }
  } else if(runMode==3) { //meshFields
    fprintf(stderr, "mode: meshFields\n");
    constexpr int extent = effMajorSize/numVerts;
    using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double[extent], double[extent]>;
     Controller c(numVerts);
     MeshField::MeshField<Controller> cabMeshField(c);
     
     auto eff_major = cabMeshField.makeField<0>();
     auto eff_minor = cabMeshField.makeField<1>();

     for (int i = 0; i < numIter; i++) {
       gyroScatterMeshFields(e_half, fmap_d, bmap_d,
			     fweights_d, bweights_d,
			     eff_major, eff_minor,
			     cabMeshField,
			     numRings, numPtsPerRing,
			     owners_d, std::string("MeshFields"));
     }
  }
  else if(runMode==4){ //kokkos view
    fprintf(stderr, "mode: kokkosView\n");
    /*  Create 2 kokkos views as eff_major and eff_minor
     *  pass into gyroScatterKokkos
     *  +---------------------------------------------+
     *  | Vector01      | Vector02      | Vector03    | ...
     *  +---------------+-------------+---------------+
     *  | vtx01 | vtx02 | vtx03 | vtx04 | vtx05 | vtx06 | ...
     *  +-------+-------+-------+-------+-------+-------+
     *  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c| ...
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *   ^ ^
     *   | |__________
     *   |            |
     *  +-----------------------------+
     *  |Component|Component|Component| ...
     *  +---------+---------+---------+
     *  | r0 | r1 | r0 | r1 | r0 | r1 | ...
     *  +----+----+----+----+----+----+
     *  
     *
     *
     *  -- fit as many 'extents' inside the vector size.
     *  -- generate appropriate ranges for vector loop variables.
     *
     *  cabana: eff_major.access(s,a,i*gnrp1+ring)
     *  -> eff_major[ (stride * s) + a + (VectorLength * (i*gnrp1+ring)) ];
     *
     *  we know that vtx = s*VectorLength+a
     *  and vtx must vary between [0,numVerts)
     *    
     * 
     */
    
    constexpr int extent = effMajorSize/numVerts;
    auto majorNumSOA = std::floor( numVerts/VectorLength );
    auto minorNumSOA = std::floor( numVerts/VectorLength );

    if( effMajorSize % VectorLength > 0 ) majorNumSOA++;
    if( effMinorSize % VectorLength > 0 ) minorNumSOA++;
    
    Kokkos::View<double*, MemorySpace> eff_major( "eff_major", majorNumSOA*extent*VectorLength );
    Kokkos::View<double*, MemorySpace> eff_minor( "eff_major", minorNumSOA*extent*VectorLength );
      
    for( int i = 0; i < numIter; i++ )
    {
        gyroScatterKokkos( e_half, fmap_d, bmap_d,
                            fweights_d, bweights_d,
                            eff_major, eff_minor,
                            numRings, numPtsPerRing,
                            owners_d, majorNumSOA, minorNumSOA );
    }
    

  } else {
    fprintf(stderr, "Error: invalid run mode (must be 0, 1, 2, 3, or 4)\n");
    return 0;
  }

  fprintf(stderr, "done\n");
  return 0;
}
