#include <complex>
#include <string>
#include <cuda_runtime.h>

#include "analysis.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

int main(int argc, char *argv[])
{
  using T = std::complex<double>;
  using Basis = TetrahedralBasis;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  // Load in the mesh
  std::string filename("input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  // Set the number of degrees of freeom
  int ndof = 3 * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *direction = new T[ndof];
  double *p = new double[ndof];
  double h = 1e-30;
  for (int i = 0; i < ndof; i++)
  {
    direction[i] = (double)rand() / RAND_MAX;
    p[i] = (double)rand() / RAND_MAX;
    dof[i] = 0.01 * rand() / RAND_MAX;
    dof[i] += T(0.0, h * direction[i].real());
    res[i] = 0.0;
    Jp[i] = 0.0;
  }

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;

  // Allocate space for the residual
  T energy = Analysis::energy(num_elements, element_nodes, xloc, dof, num_nodes, C1, D1);
  Analysis::residual(num_elements, num_nodes, element_nodes, xloc, dof, res, C1, D1);
  Analysis::jacobian_product(num_elements, element_nodes, xloc, dof,
                             direction, Jp, C1, D1);

  std::cout << energy << std::endl;

  double dres_cs = energy.imag() / h;
  double dres_exact = 0.0;
  double dJp_cs = 0.0;
  double dJp_exact = 0.0;
  for (int i = 0; i < ndof; i++)
  {
    dres_exact += res[i].real() * direction[i].real();
    dJp_cs += p[i] * res[i].imag() / h;
    dJp_exact += Jp[i].real() * p[i];
  }

  std::printf("\nDerivatives check for the residual\n");
  std::printf("complex step derivatives: %25.15e\n", dres_cs);
  std::printf("exact derivatives:        %25.15e\n", dres_exact);
  std::printf("relative error:           %25.15e\n",
              (dres_exact - dres_cs) / dres_cs);

  std::printf("\nDerivatives check for the Jacobian-vector product\n");
  std::printf("complex step derivatives: %25.15e\n", dJp_cs);
  std::printf("exact derivatives:        %25.15e\n", dJp_exact);
  std::printf("relative error:           %25.15e\n",
              (dJp_exact - dJp_cs) / dJp_cs);

  return 0;
}