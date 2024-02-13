template <int ndof, typename T, int spatial_dim, int nodes_per_element>
__device__ static void add_element_res(const int nodes[], const T element_res[],
                                       T res[])
{
  for (int j = 0; j < nodes_per_element; j++)
  {
    int node = nodes[j];
    for (int k = 0; k < spatial_dim; k++, element_res++)
    {
      atomicAdd(res[ndof * node + k], element_res[0]);
    }
  }
}

// template <typename T>
// __device__ static void eval_basis_grad(const T pt[], T Nxi[])
// {
//   // Corner node derivatives
//   Nxi[0] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
//   Nxi[1] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
//   Nxi[2] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
//   Nxi[3] = 4.0 * pt[0] - 1.0;
//   Nxi[4] = 0.0;
//   Nxi[5] = 0.0;
//   Nxi[6] = 0.0;
//   Nxi[7] = 4.0 * pt[1] - 1.0;
//   Nxi[8] = 0.0;
//   Nxi[9] = 0.0;
//   Nxi[10] = 0.0;
//   Nxi[11] = 4.0 * pt[2] - 1.0;

//   // Mid node derivatives
//   Nxi[12] = -4.0 * (2.0 * pt[0] + pt[1] + pt[2] - 1.0);
//   Nxi[13] = -4.0 * pt[0];
//   Nxi[14] = -4.0 * pt[0];

//   Nxi[15] = 4.0 * pt[1];
//   Nxi[16] = 4.0 * pt[0];
//   Nxi[17] = 0.0;

//   Nxi[18] = -4.0 * pt[1];
//   Nxi[19] = -4.0 * (pt[0] + 2.0 * pt[1] + pt[2] - 1.0);
//   Nxi[20] = -4.0 * pt[1];

//   Nxi[21] = -4.0 * pt[2];
//   Nxi[22] = -4.0 * pt[2];
//   Nxi[23] = -4.0 * (pt[0] + pt[1] + 2.0 * pt[2] - 1.0);

//   Nxi[24] = 4.0 * pt[2];
//   Nxi[25] = 0.0;
//   Nxi[26] = 4.0 * pt[0];

//   Nxi[27] = 0.0;
//   Nxi[28] = 4.0 * pt[2];
//   Nxi[29] = 4.0 * pt[1];
// }

template <typename T, int spatial_dim, int nodes_per_element>
__device__ static void add_grad(const T pt[], const T coef[], T res[])
{
  T Nxi[spatial_dim * nodes_per_element];
  eval_basis_grad(pt, Nxi);

  for (int i = 0; i < nodes_per_element; i++)
  {
    for (int k = 0; k < spatial_dim; k++)
    {
      res[spatial_dim * i + k] +=
          (coef[spatial_dim * k] * Nxi[spatial_dim * i] +
           coef[spatial_dim * k + 1] * Nxi[spatial_dim * i + 1] +
           coef[spatial_dim * k + 2] * Nxi[spatial_dim * i + 2]);
    }
  }
}

template <typename T>
__device__ inline void mat3x3MatTransMult(const T A[], const T B[], T C[])
{
  C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
  C[3] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
  C[6] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];

  C[1] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
  C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
  C[7] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];

  C[2] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
  C[5] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
  C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}

template <typename T>
__device__ inline void addDet3x3Sens(const T s, const T A[], T Ad[])
{
  Ad[0] += s * (A[8] * A[4] - A[7] * A[5]);
  Ad[1] += s * (A[6] * A[5] - A[8] * A[3]);
  Ad[2] += s * (A[7] * A[3] - A[6] * A[4]);

  Ad[3] += s * (A[7] * A[2] - A[8] * A[1]);
  Ad[4] += s * (A[8] * A[0] - A[6] * A[2]);
  Ad[5] += s * (A[6] * A[1] - A[7] * A[0]);

  Ad[6] += s * (A[1] * A[5] - A[2] * A[4]);
  Ad[7] += s * (A[3] * A[2] - A[0] * A[5]);
  Ad[8] += s * (A[0] * A[4] - A[3] * A[1]);
}

template <typename T>
__device__ inline void det3x32ndSens(const T s, const T A[], T Ad[])
{
  // Ad[0] = s*(A[8]*A[4] - A[7]*A[5]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = s * A[8];
  Ad[5] = -s * A[7];
  Ad[6] = 0.0;
  Ad[7] = -s * A[5];
  Ad[8] = s * A[4];
  Ad += 9;

  // Ad[1] += s*(A[6]*A[5] - A[8]*A[3]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = -s * A[8];
  Ad[4] = 0.0;
  Ad[5] = s * A[6];
  Ad[6] = s * A[5];
  Ad[7] = 0.0;
  Ad[8] = -s * A[3];
  ;
  Ad += 9;

  // Ad[2] += s*(A[7]*A[3] - A[6]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = s * A[7];
  Ad[4] = -s * A[6];
  Ad[5] = 0.0;
  Ad[6] = -s * A[4];
  Ad[7] = s * A[3];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[3] += s*(A[7]*A[2] - A[8]*A[1]);
  Ad[0] = 0.0;
  Ad[1] = -s * A[8];
  Ad[2] = s * A[7];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = s * A[2];
  Ad[8] = -s * A[1];
  Ad += 9;

  // Ad[4] += s*(A[8]*A[0] - A[6]*A[2]);
  Ad[0] = s * A[8];
  Ad[1] = 0.0;
  Ad[2] = -s * A[6];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = -s * A[2];
  Ad[7] = 0.0;
  Ad[8] = s * A[0];
  Ad += 9;

  // Ad[5] += s*(A[6]*A[1] - A[7]*A[0]);
  Ad[0] = -s * A[7];
  Ad[1] = s * A[6];
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = s * A[1];
  Ad[7] = -s * A[0];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[6] += s*(A[1]*A[5] - A[2]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = s * A[5];
  Ad[2] = -s * A[4];
  Ad[3] = 0.0;
  Ad[4] = -s * A[2];
  Ad[5] = s * A[1];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[7] += s*(A[3]*A[2] - A[0]*A[5]);
  Ad[0] = -s * A[5];
  Ad[1] = 0.0;
  Ad[2] = s * A[3];
  Ad[3] = s * A[2];
  Ad[4] = 0.0;
  Ad[5] = -s * A[0];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[8] += s*(A[0]*A[4] - A[3]*A[1]);
  Ad[0] = s * A[4];
  Ad[1] = -s * A[3];
  Ad[2] = 0.0;
  Ad[3] = -s * A[1];
  Ad[4] = s * A[0];
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
}

template <typename T, int spatial_dim>
__device__ static void d_residual(T weight, const T J[], const T grad[], T coef[], T C1, T D1)
{
  // Compute the inverse and determinant of the Jacobian matrix
  T Jinv[spatial_dim * spatial_dim];
  T detJ = inv3x3(J, Jinv);

  // Compute the derformation gradient
  T F[spatial_dim * spatial_dim];
  mat3x3MatMult(grad, Jinv, F);
  F[0] += 1.0;
  F[4] += 1.0;
  F[8] += 1.0;

  // Compute the invariants
  T detF = det3x3(F);

  // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
  T I1 =
      (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
       F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

  // Compute the derivatives of the energy density wrt I1 and detF
  T bI1 = C1;
  T bdetF = -2.0 * C1 / detF + 2.0 * D1 * (detF - 1.0);

  // Add the contributions from the quadrature
  bI1 *= weight * detJ;
  bdetF *= weight * detJ;

  // Derivative in the physical coordinates
  T cphys[spatial_dim * spatial_dim];

  // Add dU0/dI1*dI1/dUx
  cphys[0] = 2.0 * F[0] * bI1;
  cphys[1] = 2.0 * F[1] * bI1;
  cphys[2] = 2.0 * F[2] * bI1;
  cphys[3] = 2.0 * F[3] * bI1;
  cphys[4] = 2.0 * F[4] * bI1;
  cphys[5] = 2.0 * F[5] * bI1;
  cphys[6] = 2.0 * F[6] * bI1;
  cphys[7] = 2.0 * F[7] * bI1;
  cphys[8] = 2.0 * F[8] * bI1;

  // Add dU0/dJ*dJ/dUx
  addDet3x3Sens(bdetF, F, cphys);

  // Transform back to derivatives in the computational coordinates
  mat3x3MatTransMult(cphys, Jinv, coef);
}

template <typename T, int spatial_dim, int nodes_per_element, int num_quadrature_pts>
__device__ static T compute_residual_for_element(const int element_nodes[], const T C1, const T D1, const int i, const T xloc[], const T dof[], T *res)
{
  // Get the element node locations
  T element_xloc[spatial_dim * nodes_per_element];
  get_element_dof<spatial_dim, T>(&element_nodes[nodes_per_element * i], xloc,
                                  element_xloc);
  const int dof_per_element = spatial_dim * nodes_per_element;
  // Get the element degrees of freedom
  T element_dof[dof_per_element];
  get_element_dof<spatial_dim, T>(&element_nodes[nodes_per_element * i], dof,
                                  element_dof);

  // Create the element residual
  T element_res[dof_per_element];
  for (int j = 0; j < dof_per_element; j++)
  {
    element_res[j] = 0.0;
  }

  for (int j = 0; j < num_quadrature_pts; j++)
  {
    T pt[spatial_dim];
    T weight = get_quadrature_pt<T>(j, pt);

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    T J[spatial_dim * spatial_dim];
    eval_grad<T, spatial_dim>(pt, element_xloc, J);

    // Evaluate the derivative of the dof in the computational coordinates
    T grad[spatial_dim * spatial_dim];
    eval_grad<T, spatial_dim>(pt, element_dof, grad);

    // Evaluate the residuals at the quadrature points
    T coef[spatial_dim * spatial_dim];
    d_residual<T, spatial_dim>(weight, J, grad, coef);

    // Add the contributions to the element residual
    add_grad<T, spatial_dim, nodes_per_element>(pt, coef, element_res);
  }

  add_element_res<spatial_dim, T, spatial_dim, nodes_per_element>(&element_nodes[nodes_per_element * i],
                                                                  element_res, res);
}

template <typename T, int spatial_dim, int nodes_per_element, int num_quadrature_pts>
__global__ static void residual_kernel(int *num_elements, const int element_nodes[],
                                       const T xloc[], const T dof[], T *res, const T C1, const T D1)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= *num_elements)
    return;

  compute_residual_for_element<T, spatial_dim, nodes_per_element, num_quadrature_pts>(element_nodes, C1, D1, idx, xloc, dof, res);
}

// This is a host function that sets up and launches the kernel
template <typename T>
void residual(int *num_elements, const int element_nodes[],
              const T xloc[], const T dof[], T res[])
{
  const int nodes_per_element = 10;
  const int ndof = 3;
  const T C1 = 0.01;
  const T D1 = 0.5;
  const int num_quadrature_pts = 5;
  const int spatial_dim = 3;

  T *d_residual; // Device pointer for the result

  size_t totalSize = *num_elements * nodes_per_element * ndof * sizeof(T);

  // Allocate memory for the result on the device and initialize to 0
  cudaMalloc(&d_residual, totalSize);
  cudaMemset(d_residual, 0, totalSize);
  // Calculate grid and block sizes
  // int blockSize = 512;  // placeholder
  // int gridSize = (num_elements / blockSize) + 1;

  int blockSize = 1;
  int gridSize = 1;
  printf("grid: %i \n", gridSize);
  printf("total: %i \n", gridSize * 512);
  printf("elems: %i \n", num_elements);

  // Launch the kernel
  // energy_kernel<T><<<gridSize, blockSize>>>(num_elements, element_nodes,
  // xloc, dof, d_total_energy);
  residual_kernel<T, spatial_dim, nodes_per_element, num_quadrature_pts><<<gridSize, blockSize>>>(num_elements, element_nodes,
                                                                                                  xloc, dof, d_residual, C1, D1);

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  // Copy the result back to the host
  cudaMemcpy(*res, d_residual, totalSize, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_residual);
}