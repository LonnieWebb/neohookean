#pragma once

#include "physics.h"
#include "tetrahedral.h"
#include <cuda_runtime.h>

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis;
using T = double;
using Basis = TetrahedralBasis;
using Quadrature = TetrahedralQuadrature;
using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

template <typename T>
__global__ void energy_kernel(int element_nodes[], T xloc[], T *total_energy)
{
  using Analysis = FEAnalysis<T, Basis, TetrahedralQuadrature, NeohookeanPhysics<T>>;
  int element_index = blockIdx.x;
  int thread_index = threadIdx.x;

  // TODO: pass these in
  const int nodes_per_element = 10;
  const int dof_per_node = 3;
  const int spatial_dim = 3;

  __shared__ T elem_energy;
  __shared__ T element_xloc[nodes_per_element * dof_per_node];
  elem_energy = 0.0;

  Analysis::get_element_dof<spatial_dim>(
      &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
}

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis
{
public:
  // Static data taken from the element basis
  static const int spatial_dim = 3;
  static const int nodes_per_element = Basis::nodes_per_element;

  // Static data from the qaudrature
  static const int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Static data taken from the physics
  static const int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static const int dof_per_element = dof_per_node * nodes_per_element;

  template <int ndof>
  static __device__ void get_element_dof(const int nodes[], const T dof[],
                                         T element_dof[])
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_dof++)
      {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

  template <int ndof>
  static __device__ void add_element_res(const int nodes[], const T element_res[],
                                         T res[])
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_res++)
      {
        res[ndof * node + k] += element_res[0];
      }
    }
  }
  // static T energy(Physics &phys, int num_elements, const int element_nodes[],
  //                 const T xloc[], const T dof[])
  // {
  //   cudaError_t err = cudaGetLastError();
  //   T total_energy = 0.0;
  //   const int threads_per_block = num_quadrature_pts;
  //   const int num_blocks = num_elements;

  //   printf("Total Elements: %i \n", num_elements);
  //   printf("Num Blocks: %i \n", num_blocks);
  //   printf("Total Threads: %i \n", num_blocks * threads_per_block);

  //   Physics *d_phys;
  //   cudaMalloc(&d_phys, sizeof(Physics));
  //   cudaMemcpy(d_phys, &phys, sizeof(Physics), cudaMemcpyHostToDevice);

  //   energy_kernel<T, Physics, Basis, Quadrature, Anaylsis><<<num_blocks, threads_per_block>>>(d_phys, d_quad, d_anly, num_elements, element_nodes, xloc, dof);
  //   cudaDeviceSynchronize();

  //   if (err != cudaSuccess)
  //   {
  //     printf("CUDA error: %s\n", cudaGetErrorString(err));
  //   }
  //   return 0.0;
  // }

  static T energy(Physics &phys, int num_elements, const int num_nodes, const int element_nodes[],
                  const T xloc[], const T dof[])
  {
    T total_energy = 3.14;
    const int num_blocks = 1;
    const int threads_per_block = 1;

    T *d_total_energy;
    cudaMalloc(&d_total_energy, sizeof(T));
    cudaMemset(d_total_energy, 0.0, sizeof(T));

    int *d_element_nodes;
    cudaMalloc(&d_element_nodes, sizeof(int) * num_elements * nodes_per_element);
    cudaMemcpy(d_element_nodes, element_nodes, sizeof(int) * num_elements * nodes_per_element, cudaMemcpyHostToDevice);

    T *d_xloc;
    cudaMalloc(&d_xloc, sizeof(T) * num_nodes * spatial_dim);
    cudaMemcpy(d_xloc, xloc, sizeof(T) * num_nodes * spatial_dim, cudaMemcpyHostToDevice);

    energy_kernel<<<num_blocks, threads_per_block>>>(d_element_nodes, d_xloc, d_total_energy);

    return total_energy;
  }
};
using T = double;
using Basis = TetrahedralBasis;
using Quadrature = TetrahedralQuadrature;
using Physics = NeohookeanPhysics<T>;
using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

template __global__ void energy_kernel<T>(int element_nodes[], T xloc[], T *total_energy);