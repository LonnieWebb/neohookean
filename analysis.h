#pragma once

#include "tetrahedral.h"
#include "physics.h"

namespace FEAnalysis
{
  using namespace NeohookeanPhysics;
  using namespace TetrahedralBasis;
  using namespace TetrahedralQuadrature;

  // Included in other files: num_quadrature_points, nodes_per_element, spatial_dim, dof_per_node
  inline constexpr int dof_per_element = spatial_dim * nodes_per_element;

  template <int ndof, typename T>
  __device__ void get_element_dof(const int nodes[], const T dof[],
                                  T element_dof[])
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < spatial_dim; k++, element_dof++)
      {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

  template <int ndof, typename T>
  __device__ static void add_element_res(const int nodes[], const T element_res[],
                                         T *res)
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < spatial_dim; k++, element_res++)
      {
        atomicAdd(&res[ndof * node + k], element_res[0]);
      }
    }
  }

  template <typename T>
  __global__ void energy_kernel(const int *element_nodes,
                                const T *xloc, const T *dof, T *total_energy, T *C1, T *D1)
  {
    int element_index = blockIdx.x;
    int thread_index = threadIdx.x;

    __shared__ T elem_energy;
    elem_energy = 0.0;
    const int dof_per_element = dof_per_node * nodes_per_element;

    __shared__ T element_xloc[dof_per_element];
    __shared__ T element_dof[dof_per_element];

    // Get the element node locations
    if (thread_index == 0)
    {

      get_element_dof<spatial_dim, T>(
          &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
      // printf("test 3 \n");
      // Get the element degrees of freedom

      get_element_dof<spatial_dim, T>(
          &element_nodes[nodes_per_element * element_index], dof, element_dof);
    }

    T pt[spatial_dim];
    T weight = get_quadrature_pt<T>(thread_index, pt);

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    T J[spatial_dim * spatial_dim];
    eval_grad<T>(pt, element_xloc, J);

    // Evaluate the derivative of the dof in the computational coordinates
    T grad[spatial_dim * spatial_dim];
    eval_grad<T>(pt, element_dof, grad);
    // Add the energy contributions
    __syncthreads();
    atomicAdd(&elem_energy, energy<T>(weight, J, grad, *C1, *D1));
    __syncthreads();
    if (thread_index == 0)
    {
      // printf("block %i, quad %i, energy %f, grad %f, element_dof %f  \n",
      //        element_index, j, elem_energy, grad[0], element_dof[0]);
      atomicAdd(total_energy, elem_energy);
    }
    if (element_index == 27601 && thread_index == 0)
    {
      for (size_t i = 0; i < 30; i++)
      {
        printf("element_dof: %f \n", element_dof[i]);
      }
    }
  }

  template <typename T>
  static T body_energy(int num_elements, int element_nodes[], T xloc[], T dof[], const int num_nodes, T C1, T D1)
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

    energy_kernel<<<num_blocks, threads_per_block>>>(d_element_nodes,
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

  template <typename T>
  __global__ void residual_kernel(int element_nodes[], const T xloc[], const T dof[],
                                  T res[], T *C1, T *D1)
  {

    __shared__ T element_res[dof_per_element];
    __shared__ T element_xloc[dof_per_element];
    __shared__ T element_dof[dof_per_element];
    int element_index = blockIdx.x;

    // Parallel initialization of element_res
    for (int i = threadIdx.x; i < dof_per_element; i += blockDim.x)
    {
      element_res[i] = 0.0;
    }

    __syncthreads();

    // Get the element node locations
    if (threadIdx.x == 0)
    {

      get_element_dof<spatial_dim, T>(
          &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
      // printf("test 3 \n");
      // Get the element degrees of freedom

      get_element_dof<spatial_dim, T>(
          &element_nodes[nodes_per_element * element_index], dof, element_dof);
    }

    __syncthreads();

    T pt[spatial_dim];
    T weight = get_quadrature_pt<T>(threadIdx.x, pt);

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    T J[spatial_dim * spatial_dim];
    eval_grad<T>(pt, element_xloc, J);

    // Evaluate the derivative of the dof in the computational coordinates
    T grad[dof_per_node * spatial_dim];
    eval_grad<T>(pt, element_dof, grad);

    // Evaluate the residuals at the quadrature points
    T coef[dof_per_node * spatial_dim];
    residual(weight, J, grad, coef, *C1, *D1);
    __syncthreads();

    // Add the contributions to the element residual
    add_grad<T, dof_per_node>(pt, coef, element_res);

    __syncthreads();
    if (threadIdx.x == 0)
    {
      add_element_res<dof_per_node>(&element_nodes[nodes_per_element * element_index],
                                    element_res, res);
    }
  }

  template <typename T>
  static void body_residual(int num_elements, int num_nodes,
                            const int element_nodes[], const T xloc[], const T dof[],
                            T res[], T C1, T D1)
  {
    cudaError_t err;
    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    T *d_res;
    cudaMalloc(&d_res, num_nodes * spatial_dim * sizeof(T));
    cudaMemset(d_res, 0, num_nodes * spatial_dim * sizeof(T));

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

    residual_kernel<<<num_blocks, threads_per_block>>>(d_element_nodes, d_xloc, d_dof,
                                                       d_res, d_C1, d_D1);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, num_nodes * spatial_dim * sizeof(T), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_element_nodes);
    cudaFree(d_xloc);
    cudaFree(d_dof);
    cudaFree(d_C1);
    cudaFree(d_D1);
    cudaFree(d_res);
  }
};