#include <cuda_runtime.h>

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
  static void get_element_dof(const int nodes[], const T dof[],
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
  static void add_element_res(const int nodes[], const T element_res[],
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
  //   T total_energy = 0.0;

  //   for (int i = 0; i < num_elements; i++)
  //   {
  //     // Get the element node locations
  //     T element_xloc[spatial_dim * nodes_per_element];
  //     get_element_dof<spatial_dim>(&element_nodes[nodes_per_element * i], xloc,
  //                                  element_xloc);

  //     // Get the element degrees of freedom
  //     T element_dof[dof_per_element];
  //     get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i], dof,
  //                                   element_dof);

  //     for (int j = 0; j < num_quadrature_pts; j++)
  //     {
  //       T pt[spatial_dim];
  //       T weight = Quadrature::template get_quadrature_pt<T>(j, pt);

  //       // Evaluate the derivative of the spatial dof in the computational
  //       // coordinates
  //       T J[spatial_dim * spatial_dim];
  //       Basis::template eval_grad<T, spatial_dim>(pt, element_xloc, J);

  //       // Evaluate the derivative of the dof in the computational coordinates
  //       T grad[dof_per_node * spatial_dim];
  //       Basis::template eval_grad<T, dof_per_node>(pt, element_dof, grad);

  //       // Add the energy contributions
  //       total_energy += phys.energy(weight, J, grad);
  //     }
  //   }

  //   return total_energy;
  // }

  static T energy(Physics &phys, int num_elements, const int element_nodes[],
                  const T xloc[], const T dof[])
  {
    cudaError_t err = cudaGetLastError();
    T total_energy = 0.0;
    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    printf("Total Elements: %i \n", num_elements);
    printf("Num Blocks: %i \n", num_blocks);
    printf("Total Threads: %i \n", num_blocks * threads_per_block);

    Physics *d_phys;
    cudaMalloc(&d_phys, sizeof(Physics));
    cudaMemcpy(d_phys, &phys, sizeof(Physics), cudaMemcpyHostToDevice);

    energy_kernel<<<num_blocks, threads_per_block>>>(d_phys, num_elements, element_nodes, xloc, dof);
    cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    return 0.0;
  }

  // static void residual(Physics& phys, int num_elements,
  //                      const int element_nodes[], const T xloc[], const T dof[],
  //                      T res[]) {
  //   for (int i = 0; i < num_elements; i++) {
  //     // Get the element node locations
  //     T element_xloc[spatial_dim * nodes_per_element];
  //     get_element_dof<spatial_dim>(&element_nodes[nodes_per_element * i], xloc,
  //                                  element_xloc);

  //     // Get the element degrees of freedom
  //     T element_dof[dof_per_element];
  //     get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i], dof,
  //                                   element_dof);

  //     // Create the element residual
  //     T element_res[dof_per_element];
  //     for (int j = 0; j < dof_per_element; j++) {
  //       element_res[j] = 0.0;
  //     }

  //     for (int j = 0; j < num_quadrature_pts; j++) {
  //       T pt[spatial_dim];
  //       T weight = Quadrature::template get_quadrature_pt<T>(j, pt);

  //       // Evaluate the derivative of the spatial dof in the computational
  //       // coordinates
  //       T J[spatial_dim * spatial_dim];
  //       Basis::template eval_grad<T, spatial_dim>(pt, element_xloc, J);

  //       // Evaluate the derivative of the dof in the computational coordinates
  //       T grad[dof_per_node * spatial_dim];
  //       Basis::template eval_grad<T, dof_per_node>(pt, element_dof, grad);

  //       // Evaluate the residuals at the quadrature points
  //       T coef[dof_per_node * spatial_dim];
  //       phys.residual(weight, J, grad, coef);

  //       // Add the contributions to the element residual
  //       Basis::template add_grad<T, dof_per_node>(pt, coef, element_res);
  //     }

  //     add_element_res<dof_per_node>(&element_nodes[nodes_per_element * i],
  //                                   element_res, res);
  //   }
  // }

  static void residual(Physics &phys, int num_elements,
                       const int element_nodes[], const T xloc[], const T dof[],
                       T res[])
  {
    // Set up parallel operations such that there's one block per elemnent and one thread per quadrature point
    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    Physics *d_phys;
    cudaMalloc(&d_phys, sizeof(Physics));
    cudaMemcpy(d_phys, &phys, sizeof(Physics), cudaMemcpyHostToDevice);

    // residual_kernel<T, Physics><<<num_blocks, threads_per_block>>>();
  }

  static void jacobian_product(Physics &phys, int num_elements,
                               const int element_nodes[], const T xloc[],
                               const T dof[], const T direct[], T res[])
  {
    for (int i = 0; i < num_elements; i++)
    {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_dof<spatial_dim>(&element_nodes[nodes_per_element * i], xloc,
                                   element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i], dof,
                                    element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[dof_per_element];
      get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i],
                                    direct, element_direct);

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++)
      {
        element_res[j] = 0.0;
      }

      for (int j = 0; j < num_quadrature_pts; j++)
      {
        T pt[spatial_dim];
        T weight = Quadrature::template get_quadrature_pt<T>(j, pt);

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<T, spatial_dim>(pt, element_xloc, J);

        // Evaluate the derivative of the dof in the computational coordinates
        T grad[dof_per_node * spatial_dim];
        Basis::template eval_grad<T, dof_per_node>(pt, element_dof, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        T grad_direct[dof_per_node * spatial_dim];
        Basis::template eval_grad<T, dof_per_node>(pt, element_direct,
                                                   grad_direct);

        // Evaluate the residuals at the quadrature points
        T coef[dof_per_node * spatial_dim];
        phys.jacobian(weight, J, grad, grad_direct, coef);

        // Add the contributions to the element residual
        Basis::template add_grad<T, dof_per_node>(pt, coef, element_res);
      }

      add_element_res<dof_per_node>(&element_nodes[nodes_per_element * i],
                                    element_res, res);
    }
  }
};

template <typename T, class Physics>
__global__ void residual_kernel(Physics *phys, int num_elements,
                                const int element_nodes[], const T xloc[], const T dof[],
                                T res[])
{
}

template <typename T, class Physics>
__global__ void energy_kernel(Physics *phys, int num_elements, const int element_nodes[],
                              const T xloc[], const T dof[])
{
  // const int dof_per_element = spatial_dim * nodes_per_element;
  // const int dof_per_node = spatial_dim;

  // __shared__ T element_xloc[spatial_dim * nodes_per_element];
  // __shared__ T element_dof[dof_per_element];

  // int element_idx = blockIdx.x;
  // if (threadIdx.x == 0)
  // {
  //   // Get the element node locations
  //   get_element_dof<spatial_dim>(&element_nodes[nodes_per_element * element_idx], xloc,
  //                                element_xloc);

  //   // Get the element degrees of freedom
  //   get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * element_idx], dof,
  //                                 element_dof);
  // }
  __syncthreads();
}