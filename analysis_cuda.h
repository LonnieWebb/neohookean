// Assuming T, Physics, Basis, Quadrature classes are defined globally or
// included before this definition
template <int ndof, typename T>
__device__ void get_element_dof(const int nodes[], const T dof[],
                                T element_dof[], const int nodes_per_element,
                                const int spatial_dim)
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

template <typename T>
__device__ static T get_quadrature_pt(int k, T pt[])
{
    if (k == 0)
    {
        pt[0] = 0.25;
        pt[1] = 0.25;
        pt[2] = 0.25;
        return -2.0 / 15;
    }
    else if (k == 1)
    {
        pt[0] = 1.0 / 6.0;
        pt[1] = 1.0 / 6.0;
        pt[2] = 1.0 / 6.0;
        return 3.0 / 40;
    }
    else if (k == 2)
    {
        pt[0] = 0.5;
        pt[1] = 1.0 / 6.0;
        pt[2] = 1.0 / 6.0;
        return 3.0 / 40;
    }
    else if (k == 3)
    {
        pt[0] = 1.0 / 6.0;
        pt[1] = 0.5;
        pt[2] = 1.0 / 6.0;
        return 3.0 / 40;
    }
    else if (k == 4)
    {
        pt[0] = 1.0 / 6.0;
        pt[1] = 1.0 / 6.0;
        pt[2] = 0.5;
        return 3.0 / 40;
    }
    return 0.0;
}

template <typename T>
__device__ void eval_basis_grad(const T pt[], T Nxi[])
{
    // Corner node derivatives
    Nxi[0] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
    Nxi[1] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
    Nxi[2] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
    Nxi[3] = 4.0 * pt[0] - 1.0;
    Nxi[4] = 0.0;
    Nxi[5] = 0.0;
    Nxi[6] = 0.0;
    Nxi[7] = 4.0 * pt[1] - 1.0;
    Nxi[8] = 0.0;
    Nxi[9] = 0.0;
    Nxi[10] = 0.0;
    Nxi[11] = 4.0 * pt[2] - 1.0;

    // Mid node derivatives
    Nxi[12] = -4.0 * (2.0 * pt[0] + pt[1] + pt[2] - 1.0);
    Nxi[13] = -4.0 * pt[0];
    Nxi[14] = -4.0 * pt[0];

    Nxi[15] = 4.0 * pt[1];
    Nxi[16] = 4.0 * pt[0];
    Nxi[17] = 0.0;

    Nxi[18] = -4.0 * pt[1];
    Nxi[19] = -4.0 * (pt[0] + 2.0 * pt[1] + pt[2] - 1.0);
    Nxi[20] = -4.0 * pt[1];

    Nxi[21] = -4.0 * pt[2];
    Nxi[22] = -4.0 * pt[2];
    Nxi[23] = -4.0 * (pt[0] + pt[1] + 2.0 * pt[2] - 1.0);

    Nxi[24] = 4.0 * pt[2];
    Nxi[25] = 0.0;
    Nxi[26] = 4.0 * pt[0];

    Nxi[27] = 0.0;
    Nxi[28] = 4.0 * pt[2];
    Nxi[29] = 4.0 * pt[1];
}

template <typename T, int spatial_dim, int nodes_per_element>
__device__ void eval_grad(const T pt[], const T dof[], T grad[])
{
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int k = 0; k < spatial_dim * spatial_dim; k++)
    {
        grad[k] = 0.0;
    }

    for (int i = 0; i < nodes_per_element; i++)
    {
        for (int k = 0; k < spatial_dim; k++)
        {
            grad[spatial_dim * k] += Nxi[spatial_dim * i] * dof[spatial_dim * i + k];
            grad[spatial_dim * k + 1] += Nxi[spatial_dim * i + 1] * dof[spatial_dim * i + k];
            grad[spatial_dim * k + 2] += Nxi[spatial_dim * i + 2] * dof[spatial_dim * i + k];
        }
    }
}

template <typename T>
__device__ inline void mat3x3MatMult_d(const T A[], const T B[], T C[])
{
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];

    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];

    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
__device__ inline T inv3x3_d(const T A[], T Ainv[])
{
    T det =
        (A[8] * (A[0] * A[4] - A[3] * A[1]) - A[7] * (A[0] * A[5] - A[3] * A[2]) +
         A[6] * (A[1] * A[5] - A[2] * A[4]));
    T detinv = 1.0 / det;

    Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
    Ainv[1] = -(A[1] * A[8] - A[2] * A[7]) * detinv;
    Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

    Ainv[3] = -(A[3] * A[8] - A[5] * A[6]) * detinv;
    Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
    Ainv[5] = -(A[0] * A[5] - A[2] * A[3]) * detinv;

    Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
    Ainv[7] = -(A[0] * A[7] - A[1] * A[6]) * detinv;
    Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;

    return det;
}

template <typename T>
__device__ inline T det3x3_d(const T A[])
{
    return (A[8] * (A[0] * A[4] - A[3] * A[1]) -
            A[7] * (A[0] * A[5] - A[3] * A[2]) +
            A[6] * (A[1] * A[5] - A[2] * A[4]));
}

template <typename T, int spatial_dim>
__device__ T calc_energy(T weight, const T J[], const T grad[], T C1, T D1)
{
    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3_d(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult_d(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3_d(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the energy density for the model
    T energy_density = C1 * (I1 - 3.0 - 2.0 * std::log(detF)) +
                       D1 * (detF - 1.0) * (detF - 1.0);

    return weight * detJ * energy_density;
}

// This is a device function that will be called from the kernel
template <typename T, int spatial_dim, int nodes_per_element>
__device__ T compute_energy_for_element(const int *element_nodes,
                                        const T *xloc, const T *dof,
                                        int element_index,
                                        int num_quadrature_pts, T C1, T D1)
{
    T elem_energy = 0.0;
    const int dof_per_element = spatial_dim * nodes_per_element;
    // printf("test 2 %d \n", nodes_per_element);

    // Get the element node locations
    T element_xloc[dof_per_element];
    get_element_dof<spatial_dim, T>(
        &element_nodes[nodes_per_element * element_index], xloc, element_xloc,
        nodes_per_element, spatial_dim);
    // printf("test 3 \n");
    // Get the element degrees of freedom
    T element_dof[dof_per_element];
    get_element_dof<spatial_dim, T>(
        &element_nodes[nodes_per_element * element_index], dof, element_dof,
        nodes_per_element, spatial_dim);

    for (unsigned int i = 0; i < 30; i++)
    {
        // printf("eldof: %f \n", *(element_xloc + i));
    }

    for (int j = 0; j < num_quadrature_pts; j++)
    {
        T pt[spatial_dim];
        T weight = get_quadrature_pt<T>(j, pt);

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        eval_grad<T, spatial_dim, nodes_per_element>(pt, element_xloc, J);

        // Evaluate the derivative of the dof in the computational coordinates
        T grad[spatial_dim * spatial_dim];
        eval_grad<T, spatial_dim, nodes_per_element>(
            pt, element_dof, grad);
        // Add the energy contributions

        elem_energy += calc_energy<T, spatial_dim>(weight, J, grad, C1, D1);
    }
    printf("elem_energy: %f \n", elem_energy);
    return elem_energy;
}

// This is the kernel function
template <typename T>
__global__ void energy_kernel(int num_elements,
                              const int *element_nodes, const T *xloc,
                              const T *dof, T *total_energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("i: %i \n", i);
    const int spatial_dim = 3;
    const int nodes_per_element = 10;
    const int num_quadrature_pts = 5;
    T C1 = 0.01;
    T D1 = 0.5;
    // printf("test in kernel func \n");

    if (i < num_elements)
    {
        // Compute energy for element i and add to total energy using atomicAdd
        T energy_contrib = compute_energy_for_element<T, spatial_dim, nodes_per_element>(element_nodes, xloc, dof, i, num_quadrature_pts, C1, D1);
        atomicAdd(total_energy, energy_contrib);
    }
}

// This is a host function that sets up and launches the kernel
template <typename T>
T energy(int num_elements, const int *element_nodes,
         const T *xloc, const T *dof)
{
    T total_energy = 0.0;
    T *d_total_energy; // Device pointer for the result

    // Allocate memory for the result on the device and initialize to 0
    cudaMalloc(&d_total_energy, sizeof(T));
    cudaMemset(d_total_energy, 0, sizeof(T));

    // Calculate grid and block sizes
    int blockSize = 512; // placeholder
    int gridSize = (num_elements / blockSize) + 1;
    printf("grid: %i \n", gridSize);
    printf("total: %i \n", gridSize * 512);
    printf("elems: %i \n", num_elements);

    // Launch the kernel
    // energy_kernel<T><<<gridSize, blockSize>>>(num_elements, element_nodes, xloc, dof, d_total_energy);
    energy_kernel<T><<<gridSize, blockSize>>>(num_elements, element_nodes, xloc, dof, d_total_energy);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(&total_energy, d_total_energy, sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_total_energy);

    return total_energy;
}