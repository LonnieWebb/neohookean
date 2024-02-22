#include <string>

#include "analysis.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

int main(int argc, char *argv[])
{
  using T = double;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;
  using namespace FEAnalysis;

  // Load in the mesh
  std::string filename("../input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  // Set the number of degrees of freeom
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

  // Allocate space for the residual
  T total_energy = body_energy(num_elements, element_nodes, xloc, dof, num_nodes, C1, D1);
  body_residual(num_elements, num_nodes, element_nodes, xloc, dof, res, C1, D1);
  // Analysis::jacobian_product(physics, num_elements, element_nodes, xloc, dof,
  //  direction, Jp);
  printf("num_elements: %i \n", num_elements);
  std::cout << total_energy << std::endl;

  printf("First 30 vals in residual \n");
  for (int i = 0; i < 30; i++)
  {
    std::cout << res[i] << std::endl;
  }

  return 0;
}