#include <string>

#include "analysis.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"
#include "analysis_cuda.h"

int main(int argc, char *argv[])
{
  using T = double;
  using Basis = TetrahedralBasis;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements;
  int *num_elements_shared;
  int num_nodes;
  int *element_nodes = nullptr;
  int *element_nodes_shared;
  int elements_per_node = 10; // C3D10
  T *dof_shared;
  T *xloc = nullptr;
  T *xloc_shared;

  // Load in the mesh
  std::string filename("../input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  cudaMallocManaged(&element_nodes_shared, num_elements * sizeof(int) * elements_per_node); // C3D10 elements
  cudaMallocManaged(&num_elements_shared, sizeof(int));
  cudaMallocManaged(&xloc_shared, sizeof(T) * 3 * num_nodes);

  for (unsigned int i = 0; i < num_elements * elements_per_node; i++)
  {
    *(element_nodes_shared + i) = *(element_nodes + i);
  }

  for (unsigned int i = 0; i < 3 * num_nodes; i++)
  {
    *(xloc_shared + i) = *(xloc + i);
  }

  *num_elements_shared = num_elements;

  delete element_nodes;
  delete xloc;

  // Set the number of degrees of freeom
  int ndof = 3 * num_nodes;

  cudaMallocManaged(&dof_shared, sizeof(T) * ndof);

  // Allocate space for the degrees of freeom
  // T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *direction = new T[ndof];
  for (int i = 0; i < ndof; i++)
  {
    dof_shared[i] = 0.01 * rand() / RAND_MAX;
    res[i] = 0.0;
    Jp[i] = 0.0;
    direction[i] = 1.0;
  }

  // for (unsigned int i = 0; i < 10; i++)
  // {
  //   std::cout << *(dof_shared + i) << std::endl;
  // }

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);
  // std::cout << sizeof((*dof)) << std::endl;

  // Allocate space for the residual
  T total_energy = energy<T>(num_elements, element_nodes_shared, xloc_shared, dof_shared);
  Analysis::residual(physics, num_elements_shared, element_nodes_shared, xloc_shared, dof_shared, res);
  Analysis::jacobian_product(physics, num_elements_shared, element_nodes_shared, xloc_shared, dof_shared,
                             direction, Jp);

  std::cout << total_energy << std::endl;

  return 0;
}