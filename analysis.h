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

template <typename T, const int nodes_per_element, const int dof_per_node, const int spatial_dim>
__global__ void energy_kernel(const int *element_nodes,
                              const T *xloc, const T *dof, T *total_energy, T *C1, T *D1)
{
  using Analysis = FEAnalysis<T, Basis, TetrahedralQuadrature, NeohookeanPhysics<T>>;
  int element_index = blockIdx.x;
  int thread_index = threadIdx.x;
  const int dof_per_element = dof_per_node * nodes_per_element;

  __shared__ T elem_energy;
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  elem_energy = 0.0;

  // Get the element node locations
  if (thread_index == 0)
  {

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
    // printf("test 3 \n");
    // Get the element degrees of freedom

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], dof, element_dof);
  }

  T pt[spatial_dim];
  T weight = TetrahedralQuadrature::get_quadrature_pt<T>(thread_index, pt);

  // Evaluate the derivative of the spatial dof in the computational
  // coordinates
  T J[spatial_dim * spatial_dim];
  TetrahedralBasis::eval_grad<T, spatial_dim>(pt, element_xloc, J);

  // Evaluate the derivative of the dof in the computational coordinates
  T grad[spatial_dim * spatial_dim];
  TetrahedralBasis::eval_grad<T, dof_per_node>(pt, element_dof, grad);
  // Add the energy contributions
  __syncthreads();
  atomicAdd(&elem_energy, Physics::energy(weight, J, grad, *C1, *D1));
  __syncthreads();
  if (thread_index == 0)
  {
    // printf("block %i, quad %i, energy %f, grad %f, element_dof %f  \n",
    //        element_index, j, elem_energy, grad[0], element_dof[0]);
    atomicAdd(total_energy, elem_energy);
  }
}

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis
{
public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = 3;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  // Static data from the qaudrature
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static constexpr int dof_per_element = dof_per_node * nodes_per_element;

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

  static T energy(int num_elements, int element_nodes[], T xloc[], T dof[], const int num_nodes, T C1, T D1)
  {
    cudaError_t err;
    T total_energy = 0.0;

    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    T *d_total_energy;
    cudaMalloc(&d_total_energy, sizeof(T));
    cudaMemset(d_total_energy, 0.0, sizeof(T));

    int *d_element_nodes;
    cudaMalloc(&d_element_nodes, sizeof(int) * num_elements * nodes_per_element);
    cudaMemcpy(d_element_nodes, element_nodes, sizeof(int) * num_elements * nodes_per_element, cudaMemcpyHostToDevice);

    T *d_xloc;
    cudaMalloc(&d_xloc, sizeof(T) * num_nodes * spatial_dim);
    cudaMemcpy(d_xloc, xloc, sizeof(T) * num_nodes * spatial_dim, cudaMemcpyHostToDevice);

    T *d_dof;
    cudaMalloc(&d_dof, sizeof(T) * num_nodes * dof_per_node);
    cudaMemcpy(d_dof, dof, sizeof(T) * num_nodes * dof_per_node, cudaMemcpyHostToDevice);

    T *d_C1;
    T *d_D1;
    cudaMalloc(&d_C1, sizeof(T));
    cudaMalloc(&d_D1, sizeof(T));

    cudaMemcpy(d_C1, &C1, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1, &D1, sizeof(T), cudaMemcpyHostToDevice);

    printf("Total Elements: %i \n", num_elements);
    printf("Num Blocks: %i \n", num_blocks);
    printf("Total Threads: %i \n", num_blocks * threads_per_block);

    energy_kernel<T, nodes_per_element, dof_per_node, spatial_dim><<<num_blocks, threads_per_block>>>(d_element_nodes,
                                                                                                      d_xloc, d_dof, d_total_energy, d_C1, d_D1);
    cudaDeviceSynchronize();
    cudaMemcpy(&total_energy, d_total_energy, sizeof(T), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_total_energy);
    cudaFree(d_element_nodes);
    cudaFree(d_xloc);
    cudaFree(d_dof);
    cudaFree(d_C1);
    cudaFree(d_D1);
    return total_energy;
  }
};

// explicit instantiation if needed

// using T = double;
// using Basis = TetrahedralBasis;
// using Quadrature = TetrahedralQuadrature;
// using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;
// const int nodes_per_element = Basis::nodes_per_element;

// template __global__ void energy_kernel<T, nodes_per_element>(const int *element_nodes,
//                                                              const T *xloc, const T *dof, T *total_energy, T *C1, T *D1);