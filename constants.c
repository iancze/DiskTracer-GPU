#include <stdio.h>
#include <math.h>

#include "constants.h"


// Helpful note of how to organize the constants with extern and header files
// https://stackoverflow.com/questions/1433204/how-do-i-use-extern-to-share-variables-between-source-files

double M_sun = 1.99e33; // [g]
double M_earth = 5.97219e27; // [g]
double AU = 1.4959787066e13; // [cm]
double pc = 3.0856776e18; // [cm]
double G = 6.67259e-8; // [cm3 g-1 s-2]
double kB = 1.380658e-16; // [erg K^-1] Boltzmann constant
double h = 6.6260755e-27; //erg s Planck constant
double c_ang = 2.99792458e18; // [A s^-1]
double cc = 2.99792458e10; // [cm s^-1]
double c_kms = 2.99792458e5; // [km s^-1]

// Conversion from degrees to radians
double deg = M_PI/180.; // [radians]

// Used when determining the necessary number of pixels in an image, given distance. Anything below
// 2 is not Nyquist sampled. This is currently set to 2.2 to provide a degree of oversampling.
double nyquist_factor = 2.2;

// Atomic
double amu = 1.6605402e-24; // [g]

// Molecular
double mu_gas = 2.37; // mean molecular weight of circumstellar gas
double m_H = 1.6733e-24; // g mass of hydrogen atom

double f_12CO = 7.5e-5;

// These variables will be calculated based upon f_12CO, so we're declaring them here, and computing
// their values in init_constants()
double X_12CO, X_13CO, X_C18O;

// molecular hydrogen number ratio to gas [unitless]
// [n_H2/n_gas] = 0.8
double X_H2 = 0.8;

double chi_12CO, chi_13CO, chi_C18O, m_CO, m_12CO, m_13CO, m_C18O;

struct molecule CO12_21, CO12_32, CO13_21, CO13_32, CO18_21, CO18_32;

// Simply use midpoint formula and adaptive steps to measure tau
double TAU_THRESH = 8.0;

// function to initialize molecule structures
// Takes in a pointer to the molecule structure.
void init_molecule(struct molecule * p, double X_mol, double B0, double mu, double mol_weight, double nu_0, int l, double T_L)
{
  // Access the fields of the structure
  // -> is shorthand for (*p).X_mol, etc..
  p->X_mol = X_mol;
  p->B0 = B0;
  p->mu = mu;
  p->mol_weight = mol_weight;
  p->nu_0 = nu_0;
  p->l = l;
  p->T_L = T_L;
}

void init_constants(void)
{
  // Using numbers from Charlie Qi
  // Number ratios measured relative to all single H nuclei (not H2) [unitless]
  // f_12CO = 7.5e-5;
  X_12CO = 2.0 * f_12CO;
  X_13CO = 1.0/69. * X_12CO;
  X_C18O = 1.0/557. * X_12CO;

  // Number ratios relative to average molecule (chi_CO)
  // Multiply this against rho to get number of molecules
  chi_12CO = X_H2 * X_12CO/(mu_gas * amu);
  chi_13CO = X_H2 * X_13CO/(mu_gas * amu);
  chi_C18O = X_H2 * X_C18O/(mu_gas * amu);

  // CO
  m_CO = 28.01 * amu; //molecular weight of CO in g
  m_12CO = 27.9949 * amu; // [g]
  m_13CO = 28.9983 * amu; // [g]
  m_C18O = 29.9992 * amu; // [g]

  // Use the initialization function to fill out the individual fields of the molecule.
  init_molecule(&CO12_21, chi_12CO, 57635.96e6, 1.1011e-19, m_12CO, 230.53800000e6, 1, 5.5321);
  init_molecule(&CO12_32, chi_12CO, 57635.96e6, 1.1011e-19, m_12CO, 345.79598990e6, 2, 16.5962);
  init_molecule(&CO13_21, chi_13CO, 55101.01e6, 1.1046e-19, m_13CO, 220.39868420e6, 1, 5.2888);
  init_molecule(&CO13_32, chi_13CO, 55101.01e6, 1.1046e-19, m_13CO, 330.58796530e6, 2, 15.8662);
  init_molecule(&CO18_21, chi_C18O, 54891.42e6, 1.1079e-19, m_C18O, 219.56035410e6, 1, 5.2686);
  init_molecule(&CO18_32, chi_C18O, 54891.42e6, 1.1079e-19, m_C18O, 329.33055250e6, 2, 15.8059);

}
