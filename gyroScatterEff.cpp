#include <Omega_h_library.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <cstdlib> //exit_failure
#include <fstream> //std::ifstream
#include "gyroScatterData0.txt" //defines constants

namespace oh = Omega_h;

namespace {
  template <typename T>
  oh::Read<T> readArrayBinary(std::string fname) {
    std::ifstream inBin(fname, std::ios::binary);
    assert(inBin.is_open());
    const auto compressed = false;
    const auto needs_swapping = !oh::is_little_endian_cpu();
    fprintf(stderr, "reading %s\n", fname.c_str());
    oh::Read<T> array;
    oh::binary::read_array(inBin, array, compressed, needs_swapping);
    inBin.close();
    return array;
  }
};

void gyroScatter(oh::Reals e_half,
    oh::LOs& forward_map, oh::LOs& backward_map,
    oh::Write<oh::Real> eff_major, oh::Write<oh::Real> eff_minor,
    const oh::LO gnrp1, const oh::LO gppr,
    oh::LOs& owners) {
  const int ncomps = e_half.size() / (2 * numVerts);
  assert(ncomps == numComponents);
  int mesh_rank = 0;
  const oh::LO nvpe = numVertsPerElm;
  // handle ring = 0
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
  assert(cudaSuccess==cudaDeviceSynchronize());
//
//  // handle ring > 0
//  auto efield_scatter = OMEGA_H_LAMBDA(const int vtx) {
//    if (owners[vtx] == mesh_rank) {
//      for(int ring=1; ring < gnrp1; ring++) {
//        // index on gyro averaged electric field
//        const auto index = vtx * gnrp1 * ncomps + ring;
//        for(int pt=0; pt<gppr; pt++) {
//          for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
//            const auto mappedVtx_f = forward_map.mappedVertex(vtx, ring, pt, elmVtx);
//            const auto mappedWgt_f = forward_map.mappedWeight(vtx, ring, pt, elmVtx);
//            const auto mappedVtx_b = backward_map.mappedVertex(vtx, ring, pt, elmVtx);
//            const auto mappedWgt_b = backward_map.mappedWeight(vtx, ring, pt, elmVtx);
//
//            // Only compute contributions of owned vertices.
//            // Field Sync will sum all contributions.
//            // This part of the operation is basically a Matrix (sparse matrix)
//            // and vector multiplication: c_j = A_ij * b_j, where vector b and
//            // c are vectors defined on the mesh vertices, with b_j, c_j the
//            // value at vertex j; while A is the gyro-average mapping matrix,
//            // A_ij represents the mapping weight from vertex i to vertex j
//            // (from field vector b to field vector c). We need to make sure
//            // the index is correct in performing this operation
//            if (mappedVtx_f >= 0) {
//              for (int i = 0; i < ncomps; ++i) {
//                // access the major component of e_half
//                //TODO: atomic_add probably is not needed here
//                const oh::LO gyroVtxIdx_f = 2 * (mappedVtx_f * ncomps + i) + 1;
//                Kokkos::atomic_add(&(eff_major[index + i * gnrp1]),
//                    mappedWgt_f * e_half[gyroVtxIdx_f] / gppr);
//              }
//            }
//            if (mappedVtx_b >= 0) {
//              for (int i = 0; i < ncomps; ++i) {
//                // access the minor component of e_half
//                //TODO: atomic_add probably is not needed here
//                const oh::LO gyroVtxIdx_b = 2 * (mappedVtx_b * ncomps + i);
//                Kokkos::atomic_add(&(eff_minor[index + i * gnrp1]),
//                    mappedWgt_b * e_half[gyroVtxIdx_b] / gppr);
//              }
//            }
//          }
//        }
//      }
//    }
//  };
//  oh::parallel_for(numVerts, efield_scatter, "gyroScatterEFF");
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "Usage: %s <field prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  std::string fname(argv[1]);
  auto lib = Omega_h::Library(&argc, &argv);

  auto fmap_d = readArrayBinary<oh::LO>(fname+"_fmap.bin");
  auto bmap_d = readArrayBinary<oh::LO>(fname+"_bmap.bin");
  auto owners_d = readArrayBinary<oh::LO>(fname+"_owners.bin");

  oh::Reals e_half(ehalfSize);
  oh::Write<oh::Real> eff_major(effMajorSize);
  oh::Write<oh::Real> eff_minor(effMinorSize);

  gyroScatter(e_half, fmap_d, bmap_d,
      eff_major, eff_minor,
      numRings, numPtsPerRing,
      owners_d);
  fprintf(stderr, "done\n");
  return 0;
}
