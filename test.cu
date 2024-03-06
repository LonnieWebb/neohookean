#include <string>
#include <cuda_runtime.h>

#include "analysis.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

int main(int argc, char *argv[])
{
  using T = double;
  using Basis = TetrahedralBasis;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements, num_nodes, num_node_sets;
  int *element_nodes, *node_set_starts, *node_set_indices;
  T *xloc;
  std::vector<std::string> node_set_names;

  // Load in the mesh
  std::string filename("../input/Tensile1.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes,
               &num_node_sets, &element_nodes, &xloc,
               &node_set_starts, &node_set_indices,
               &node_set_names);

  // Set the number of degrees of freedom
  int ndof = 3 * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *direction = new T[ndof];
  for (int i = 0; i < ndof; i++)
  {
    dof[i] = 0.01 * rand() / RAND_MAX;
    res[i] = 0.0;
    Jp[i] = 0.0;
    direction[i] = 1.0;
  }

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Allocate space for the residual
  T total_energy = Analysis::energy(num_elements, element_nodes, xloc, dof, num_nodes, C1, D1);
  Analysis::residual(num_elements, num_nodes, element_nodes, xloc, dof, res, C1, D1);
  // Analysis::jacobian_product(num_elements, element_nodes, xloc, dof,
  //                            direction, Jp, C1, D1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Execution time for function: %f ms\n", milliseconds);

  printf("num_elements: %i \n", num_elements);
  std::cout << total_energy << std::endl;

  printf("First 30 vals in residual \n");
  for (int i = 0; i < 30; i++)
  {
    std::cout << res[i] << std::endl;
  }

  return 0;
}