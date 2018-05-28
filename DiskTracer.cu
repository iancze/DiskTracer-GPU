#include <stdio.h>
#include <stdbool.h>

// CUDA math library
#include "math.h"

// Necessary for writing the grid properties array to disk
#include <hdf5.h>
#include <hdf5_hl.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// Set up for the problem
// Number of pix per dimension
#define NPIX 128
// Number of velocities
#define NVEL 4

// Maximum extent of the image (in AU)
#define RMAX 800.0



// ***************************
// Constants
// ***************************

// To hold molecular constants
struct molecule {
    double X_mol; // number fraction of molecule
    double B0; // rigid rotor constant of molecule (Hz)
    double mu; // (e.s.u. cm) permanent dipole moment of the molecule
    double mol_weight; // (g) molecular weight
    double nu_0; // (Hz) central frequency of transition
    int l; // lower level of transition
    double T_L; // (K) temperature equivalent of energy at lower transition
};

// standard model parameters
struct pars{
    double M_star; // [M_sun] stellar mass
    double r_c; // [AU] characteristic radius
    double T_10; // [K] temperature at 10 AU
    double q; // temperature gradient exponent
    double gamma; // surface density gradient exponent
    double Sigma_c; // [g/cm^2] surface density at characteristic radius
    double ksi; // [cm s^{-1}] microturbulence
    double dpc; //[pc] distance to system
    double incl; // [degrees] inclination 0 deg = face on, 90 = edge on.
    double PA; // [degrees] position angle (East of North)
    double vel; // [km/s] systemic velocity (positive is redshift/receeding)
    double mu_RA; // [arcsec] central offset in RA
    double mu_DEC; //[arcsec] central offset in DEC
};

// For pre-caching geometry calculations along a ray
struct geoPrecalc{
  double xp2; // xp2 = xprime^2
  double a1; // a1 = cosd(incl) * yprime
  double a2; // a2 = sind(incl)
  double b1; // b1 = sind(incl) * yprime
  double b2; // b2 = cosd(incl)
  double c1; // c1 = sqrt(G * M_star * M_sun) * sind(incl) * xprime
  double c2; // c2 = xprime^2 + yprime^2
};

// Bounding z^prime points for tracing a ray
struct zps{
  double z1start;
  double z1end;
  double z2start;
  double z2end;
  bool merge;
};

// A set of cylindrical coordinates calculated along a ray
struct coords{
  double rcyl;
  double z;
  double vlos;
};

// definition of  Grid interpolation structure
// Hosts 2D arrays which will be allocated by malloc() upon runtime
struct grid{
  int nr; // number of radial bins (cylindrical radius)
  double rmax; // [cm] maximum radial extent of the disk (cylindrical radius)
  double dr; // [cm] the physical change in r for each index
  int nz; // number of vertical bins
  double zmax; // [cm] maximum vertical extent of the disk
  double dz; // [cm] the physical change in z for each index
  double * pDeltaV2; // (nz, nr) pointer to the DeltaV2 array
  double * pS_nu; // (nz, nr) pointer to the S_nu array
  double * pUpsilon; // (nz, nr) pointer to the Upsilon
};

// An interpolated point of DeltaV2, S_nu, and Upsilon
struct interp_point{
  double DeltaV2;
  double S_nu;
  double Upsilon;
};


double M_sun = 1.99e33; // [g]
double AU = 1.4959787066e13; // [cm]
double pc = 3.0856776e18; // [cm]
double G = 6.67259e-8; // [cm3 g-1 s-2]
double kB = 1.380658e-16; // [erg K^-1] Boltzmann constant
double h = 6.6260755e-27; //erg s Planck constant
double cc = 2.99792458e10; // [cm s^-1]

// Conversion from degrees to radians
// double deg = M_PI/180.; // [radians]

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
// void init_molecule(struct molecule * p, double X_mol, double B0, double mu, double mol_weight, double nu_0, int l, double T_L)
// {
//   // Access the fields of the structure
//   // -> is shorthand for (*p).X_mol, etc..
//   p->X_mol = X_mol;
//   p->B0 = B0;
//   p->mu = mu;
//   p->mol_weight = mol_weight;
//   p->nu_0 = nu_0;
//   p->l = l;
//   p->T_L = T_L;
// }

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
  // init_molecule(&CO12_21, chi_12CO, 57635.96e6, 1.1011e-19, m_12CO, 230.53800000e6, 1, 5.5321);
  // init_molecule(&CO12_32, chi_12CO, 57635.96e6, 1.1011e-19, m_12CO, 345.79598990e6, 2, 16.5962);
  // init_molecule(&CO13_21, chi_13CO, 55101.01e6, 1.1046e-19, m_13CO, 220.39868420e6, 1, 5.2888);
  // init_molecule(&CO13_32, chi_13CO, 55101.01e6, 1.1046e-19, m_13CO, 330.58796530e6, 2, 15.8662);
  // init_molecule(&CO18_21, chi_C18O, 54891.42e6, 1.1079e-19, m_C18O, 219.56035410e6, 1, 5.2686);
  // init_molecule(&CO18_32, chi_C18O, 54891.42e6, 1.1079e-19, m_C18O, 329.33055250e6, 2, 15.8059);

}


// ****************************************
// * model functions
// ****************************************


// Assume all inputs to these functions are in CGS units and in *cylindrical* coordinates.
// Parametric type T allows passing individual Float64 or Vectors.
// Alternate functions accept pars passed around, where pars is in M_star, AU, etc...
// The Keplerian velocity assuming non-zero thickness to the disk.
double velocity(double r, double z, double M_star)
{
  double a = r*r + z*z;
  // Calculating sqrt(G * M_star / (r*r + z*z)^(3./2)) * r;
  return sqrt(G * M_star / (a * sqrt(a))) * r;
}

// Calculate temperature in cylindrical coordinates.
double temperature(double r, double T_10, double q)
{
  return T_10 * pow(r / (10. * AU), -q);
}

double temperature_pars(double r, struct pars * p)
{
  return temperature(r, p->T_10, p->q);
}

// Scale height, calculate in cylindrical coordinates
double Hp(double r, double M_star, double T_10, double q) // inputs in cgs
{
  double temp = temperature(r, T_10, q); // [K]
  return sqrt(kB * temp * r*r*r /(mu_gas * m_H * G * M_star)); // [cm]
}

double Hp_pars(double r, struct pars * p)
{
  return Hp(r, p->M_star * M_sun, p->T_10, p->q);
}

// Calculate the gas surface density using cylindrical coordinates.
double Sigma(double r, struct pars * p)
{
  double r_c = p->r_c * AU;

  return p->Sigma_c * pow(r/r_c, -p->gamma) * exp(-pow(r/r_c, 2. -p->gamma));
}


// Delivers a gas density in g/cm^3
double rho(double r, double z, struct pars * p)
{
  double H = Hp_pars(r, p);
  double S = Sigma(r, p);

  // Calculate the density
  double dens = S/(sqrt(2. * M_PI) * H) * exp(-0.5 * (z/H) * (z/H));

  return dens;
}


// Ksi is microturbulent broadining width in units of km/s. Output of this function
// is in cm/s for RADMC (RADMC manual, eqn 7.12)
double microturbulence(double ksi)
{
  return ksi * 1.e5; // convert from km/s to cm/s
}


// Calculate the partition function for the temperature
// uses Mangum and Shirley expansion
// assumes B0 in Hz, depends on molecule
double Z_partition(double T, struct molecule * m)
{
  double nugget = (h * m->B0) / (kB * T);
  return 1./nugget + 1/3. + 1/15. * nugget + 4. * nugget*nugget / 315. + nugget*nugget*nugget / 315.;
}

// Calculate the source function. nu in Hz.
double S_nu(double r, double z, double nu, struct pars * p)
{
  double T = temperature_pars(r, p);
  return (2.0 * h * nu*nu*nu)/(cc*cc) / (exp(h * nu / (kB * T)) - 1.0);
}


// Calculate Upsilon, the opacity before the line profile
double Upsilon_nu(double r, double z, double nu, struct pars * p, struct molecule * m)
{
  double T = temperature_pars(r, p);

  // Many of the terms in Upsilon could be pre-calculated for the molecule.
  double g_l = 2.0 * m->l + 1.0;

  double n_l = m->X_mol * rho(r, z, p) * g_l * exp(-m->T_L / T) / Z_partition(T, m);

  // sigma_0
  double sigma_0 = 16.0 * M_PI * M_PI * M_PI / (h * h * cc) * (kB * m->T_L) * (m->l + 1) / (m->l * (2 * m->l + 1)) * (m->mu)*(m->mu);

  // calculate Delta V
  double DeltaV = sqrt(2.0 * kB * T / m->mol_weight + microturbulence(p->ksi)*microturbulence(p->ksi));

  return n_l * sigma_0 * cc / (DeltaV * m->nu_0 * sqrt(M_PI)) * (1.0 - exp(- h * nu / (kB * T)));
}

// Calculate (Delta V)^2. Returns in cm/s.
double DeltaV2(double r, double z, struct pars * p, struct molecule * m)
{
  double T = temperature_pars(r, p);
  return 2.0 * kB * T/m->mol_weight + (p->ksi * 1.e5) * (p->ksi * 1.e5);
}


// Functions to compute grid objects for fast interpolation.
// Because we will always be querying for the same rcyl, z values,
// Return the three quantities at once.
// Compared to the Julia version, this needs to be split up into a struct, and a function to query the struct.
// In CUDA, this will become a texture

// Store the arrays for DeltaV2, S_nu, and Upsilon_nu as arrays within the structure.

// The grid structure is defined in diskTracer.h

// // Initialize the structure. Return a pointer to the structure.
// struct grid * init_grid(struct pars * p, struct molecule * m, double nu, int nr, double rmax, int nz, double zmax)
// struct grid init_grid(struct pars * p, struct molecule * m, double nu, int nr, double rmax, int nz, double zmax)
// {
//
//   double dr = rmax / (nr - 1);
//   double dz = zmax / (nz - 1);
//
//   // Allocate space for three arrays with shape (nz, nr)
//   double * pDeltaV2 = malloc(nr * nz * sizeof(double));
//   double * pS_nu = malloc(nr * nz * sizeof(double));
//   double * pUpsilon = malloc(nr * nz * sizeof(double));
//
//
//   // Using the parameters, molecule, and frequency, fill in the values for DeltaV2, S_nu, and Upsilon
//   // Loop over all the grid cells
//   double r, z;
//
//   // Index variables. i = row, j=col. k is the stride, k = i * nr +j
//   int i, j, k;
//   int start_column = 10 - 1; // This means that all columns prior to 10 will have the same value as column 10
//
//   // We'll start at the tenth column (r)
//   // Due to the way this is stored as a 2D array, i corresponds to r
//   // and j actually corresponds to z
//   // but the r dimension is the columns, and the z dimension is the rows
//   // Actual radii, columns
//
//   for (j=0; j<nz; j++){
//     for (i=0; i<start_column; i++){
//       r = start_column * dr;
//       z = j * dz;
//       k = i + nr * j; // 1D linear index
//       // printf("k1 = %d\n", k);
//       pDeltaV2[k] = DeltaV2(r, z, p, m);
//       pS_nu[k] = S_nu(r, z, nu, p);
//       pUpsilon[k] = Upsilon_nu(r, z, nu, p, m);
//     };
//
//     for (i=start_column; i<nr; i++){
//       r = i * dr;
//       z = j * dz;
//       k = nr * j + i; // 1D linear index
//       // printf("k2 = %d\n", k);
//
//       pDeltaV2[k] = DeltaV2(r, z, p, m);
//       pS_nu[k] = S_nu(r, z, nu, p);
//       pUpsilon[k] = Upsilon_nu(r, z, nu, p, m);
//       // printf("Upsilon at %f AU, %f AU is %e\n", r/AU, z/AU, pUpsilon[k]);
//     };
//   };
//
//   // return an initialized grid structure
//   struct grid myGrid = {.nr=nr, .rmax=rmax, .dr=dr, .nz=nz, .zmax=zmax, .dz=dz, .pDeltaV2=pDeltaV2, .pS_nu=pS_nu, .pUpsilon=pUpsilon};
//   return myGrid;
// }



// ****************************************
// * geometry functions
// ****************************************


// Precalculate the quantities necessary to make repeated calls to `get_r_cyl`, `get_z`, and `get_vlos` efficient.
// struct geoPrecalc get_geoPrecalc(double xprime, double yprime, struct pars * p)
// {
//
//   // Initialize an empty struct
//   struct geoPrecalc temp = {};
//
//   // For get_r_cyl
//   temp.xp2 = xprime * xprime;
//   temp.a1 = cos(p->incl * deg) * yprime;
//   temp.a2 = sin(p->incl * deg);
//   // r_cyl = sqrt(xp2 + (a1 - a2 * zprime)^2)
//
//   // For get_z
//   temp.b1 = sin(p->incl * deg) * yprime;
//   temp.b2 = cos(p->incl * deg);
//   // z = b1 + b2 * zprime
//
//   // For get_vlos
//   temp.c1 = sqrt(G * p->M_star * M_sun) * sin(p->incl * deg) * xprime;
//   temp.c2 = xprime * xprime + yprime * yprime;
//   // vlos = c1 /(c2 + zprime^2)^(3/4.)
//
//   return temp;
// }

// Take a cartesian point in the sky plane and get the cylindrical radius of the disk
// Useful for querying the disk structure or velocity field.
// assumes i is in degrees
// double get_r_cyl(double xprime, double yprime, double zprime, double i)
// {
//   double temp = (cos(i * deg) * yprime - sin(i * deg) * zprime);
//   return sqrt(xprime * xprime + temp*temp);
// }

// Take a cartesian point in the sky plane and get the cylindrical z point in the disk
// assumes i is in degrees
// double get_z(double xprime, double yprime, double zprime, double i)
// {
//   return sin(i * deg) * yprime + cos(i * deg) * zprime;
// }

// Take a cartesian point in the sky plane and get the line of sight velocity.
// Negative velocity implies a blueshift (towards the observer).
// double get_vlos(double xprime, double yprime, double zprime, struct pars * p)
// {
//   // According to NVIDIA best practices guide, pg. 50, k^(3/4) can be rewritten as r = sqrt(k); r = r * sqrt(r)
//   double temp, k = xprime*xprime + yprime*yprime + zprime*zprime;
//   temp = sqrt(k);
//   temp = temp * sqrt(temp);
//   // temp = (xprime*xprime + yprime*yprime + zprime*zprime)^(3/4.)
//   return sqrt(G * p->M_star * M_sun) * sin(p->incl * deg) * xprime / temp;
//
// }

// get_coords is a function to return the necessary rcyl, z, vlos quickly along a given ray.
// struct coords get_coords(double zprime, struct geoPrecalc gcalc)
// {
//   // Empty temp struct
//   struct coords temp = {};
//
//   double t_rcyl = (gcalc.a1 - gcalc.a2 * zprime);
//   temp.rcyl = sqrt(gcalc.xp2 + t_rcyl*t_rcyl);
//
//   temp.z = gcalc.b1 + gcalc.b2 * zprime;
//
//   //vlos = gcalc.c1 / (gcalc.c2 + zprime^2)^(3./4);
//   // k^(3/4) can be rewritten as r = sqrt(k); r = r * sqrt(r)
//   double t_vlos =  sqrt(gcalc.c2 + zprime*zprime);
//   t_vlos = t_vlos * sqrt(t_vlos);
//   temp.vlos = gcalc.c1 / t_vlos;
//
//   return temp;
// }

// Return true if the pixel should be traced (or at least not immediately rejected).
// Valid for all pixels with this same xprime value.
// bool verify_pixel_x(double xprime, struct pars * p, double v0, double DeltaVmax)
// {
//   double vb_min = (v0 - 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s
//   double vb_max = (v0 + 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s
//
//   return ((xprime * sin(p->incl * deg)/vb_min) >= 0.0) || ((xprime * sin(p->incl * deg)/vb_max) >= 0.0);
// }

// Verify whether a given pixel has any any emitting regions. Assumes xprime, yprime, and rmax are in cm. The velocity
// equivalent to the frequency to be traced is given by v0 [km/s]. DeltaVmax is also in km/s.
//
// rmax2 is the maximum disk radius squared, in [cm^2].
// Returns a bool true/false
// __device__ bool verify_pixel(double xprime, double yprime, struct pars * p, double v0, double DeltaVmax, double rmax2)
// {
//   double vb_min = (v0 - 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s
//   double vb_max = (v0 + 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s
//
//   double rho2 = xprime*xprime + yprime*yprime;
//
//   if (rho2 > rmax2) return false;
//
//   bool overlap = (vb_min < 0.0) & (vb_max > 0.0);
//
//   if (xprime > 0.0) {
//       if (vb_max < 0.0)
//         return false;
//       else if (overlap)
//         return true;
//       else {
//         // Calculate (xprime * sqrt(G * pars.M_star * M_sun) * sin(p.incl * deg) / vb_min)^(4/3)
//         double temp = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * deg) / vb_min;
//         // k^(4/3) can be written as r = x * cbrt(x)
//         return rho2 <= (temp * cbrt(temp));
//       }
//   }
//   else if (xprime < 0.0) {
//       if (vb_min > 0.0)
//         return false;
//       else if (overlap)
//         return true;
//       else {
//         // Calculate (xprime * sqrt(G * pars.M_star * M_sun) * sin(p.incl * deg) / vb_max)^(4/3)
//         double temp = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * deg) / vb_max;
//         // k^(4/3) can be written as r = x * cbrt(x)
//         return rho2 <= (temp * cbrt(temp));
//       }
//   }
//   else return true;
// }


// Helper function to better calculate the 4/3 power (four thirds power => ftp) and reduce clutter
// double ftp(double t, double v)
// {
//   // (t/v)^(4/3.)
//   // according to NVIDIA developers guide, this is best carried out by
//   // r = x * cbrt(x);
//   double temp = t/v;
//   return temp * cbrt(temp);
// }


// Get the starting and ending bounding regions on zprime, based only upon the kinematic/geometrical constraints.
// Assumes xprime, yprime, and rmax are in cm. The velocity equivalent to the frequency to be traced is given by v0
// [km/s]. DeltaVmax is the maximum velocity expected along the ray, also in km/s.
// struct zps get_bounding_zps(double xprime, double yprime, struct pars * p, double v0, double DeltaVmax, double rmax)
// {
//
//   // Initialize all to 0
//   struct zps myZps = {};
//
//   // The three-sigma bounds on the line profile
//   double vb_min = (v0 - 3.0 * DeltaVmax) * 1.e5; // convert from km/s to cm/s
//   double vb_max = (v0 + 3.0 * DeltaVmax) * 1.e5; // convert from km/s to cm/s
//
//   // Basically, we can't be larger or smaller than this.
//   double v_temp = xprime*xprime + yprime*yprime;
//   // to find x^(3/4) do r = sqrt(x); r = r * sqrt(r)
//   v_temp = sqrt(v_temp);
//   v_temp = v_temp * sqrt(v_temp);
//   // v_temp = (xprime^2 + yprime^2)^(3./4)
//   double vb_crit = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * deg)/v_temp;
//
//   // We want to assert that this pixel has already fulfilled the zeroth order check that there will be emission
//   // @assert (((xprime >= 0.0) & (vb_max >= 0.0)) | ((xprime <= 0.0) & (vb_min <= 0.0))) "Pixel $xprime, $yprime, will have no emission."
//
//   // There exists a vb=0 velocity, so the best we can say is that z1start and z2end starts and end at (rmax, -rmax),
//   // respectively
//   bool overlap = (vb_min <= 0.0) && (vb_max >= 0.0);
//
//   // The velocity where the ray intersects the plane of the sky (z^\prime = 0) exists between vb_min and vb_max
//   // This means that the two separate bounding regions merge into one.
//   bool overlap_crit = (vb_crit > vb_min) && (vb_crit < vb_max);
//
//   double xxyy = xprime*xprime + yprime*yprime;
//   double t1 = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * deg);
//
//   if ((xprime >= 0.0) & (vb_max >= 0.0))
//   {
//     if (overlap)
//     {
//       myZps.z1start = rmax;
//       myZps.z2end = -rmax;
//     }
//     else if (vb_min >= 0.0)
//     {
//       myZps.z1start = sqrt(ftp(t1, vb_min) - xxyy);
//       myZps.z2end = -sqrt(ftp(t1, vb_min) - xxyy);
//     }
//     if (overlap_crit)
//     {
//       // There exists a vb within the range of vbs which yields zprime = 0, so the two regions merge.
//       myZps.z1end = 0.0;
//       myZps.z2start = 0.0;
//       myZps.merge = true;
//       return myZps;
//     }
//     else
//     {
//       myZps.z1end = sqrt(ftp(t1, vb_max) - xxyy);
//       myZps.z2start = -sqrt(ftp(t1, vb_max) - xxyy);
//     }
//   }
//   else if ((xprime <= 0.0) & (vb_min <= 0.0))
//   {
//     if (overlap)
//     {
//       myZps.z1start = rmax;
//       myZps.z2end = -rmax;
//     }
//     else if (vb_max <= 0.0)
//     {
//       myZps.z1start = sqrt(ftp(t1, vb_max) - xxyy);
//       myZps.z2end = -sqrt(ftp(t1, vb_max) - xxyy);
//     }
//
//     if (overlap_crit)
//     {
//       // There exists a vb within the range of vbs which yields zprime = 0, so the two regions merge.
//       myZps.z1end = 0.0;
//       myZps.z2start = 0.0;
//       myZps.merge = true;
//       return myZps;
//     }
//     else
//     {
//       myZps.z1end = sqrt(ftp(t1, vb_min) - xxyy);
//       myZps.z2start = -sqrt(ftp(t1, vb_min) - xxyy);
//     }
//   }
//   // Return all 4, initialized.
//   myZps.merge = false;
//   return myZps;
// }


// ****************************************
// * Trace functions
// ****************************************



// Adaptive stepper using Midpoint Method. https://en.wikipedia.org/wiki/Midpoint_method
// void integrate_tau(double zstart, double zend, double v0, struct pars * p, struct grid * g, struct geoPrecalc gpre, double h_tau, double tau_start, double intensity_start, double max_ds, double * end_tau, double * end_I)
// {
//
//   // @assert max_ds < 0 "max_ds must be a negative number, since the ray is traced along -z"
//
//   double tot_intensity = intensity_start;
//   double zp = zstart;
//   double tau = tau_start;
//
//   // Write in coordinate conversions for querying alpha and sfunc
//   struct coords myCoords = get_coords(zp, gpre);
//
//
//   // Calculate Delta v at this position
//   double Deltav = v0 * 1.e5 - myCoords.vlos;
//
//   // Look up RT quantities from nearest-neighbor interp.
//   struct interp_point point = interp_grid(myCoords.rcyl, myCoords.z, g);
//
//   // Evaluate alpha at current zstart position
//   double alpha = point.Upsilon * exp(-Deltav*Deltav/point.DeltaV2);
//
//   // Midpoint
//   double zp2 = 0.0;
//   double h;
//
//   double expOld = exp(-tau);
//   double expNew = exp(-tau);
//
//   while ((tau < TAU_THRESH) && (zp > zend))
//   {
//       // Based upon current alpha value, calculate how much of a dz we would need to get dz * alpha = h_tau
//       // Because these are negative numbers, the smaller step is actually the maximum (closer to 0)
//       h = fmax(-h_tau / alpha, max_ds);
//
//       // Midpoint step position
//       zp2 = zp + 0.5 * h;
//
//       // Get the necessary coordinates for querying the grid.
//       myCoords = get_coords(zp2, gpre);
//
//       // Calculate Delta v at this position
//       Deltav = v0 * 1.e5 - myCoords.vlos;
//
//       // Look up RT quantities from nearest-neighbor interp, evaluated at the midpoint
//       point = interp_grid(myCoords.rcyl, myCoords.z, g);
//
//       alpha = point.Upsilon * exp(-Deltav*Deltav/point.DeltaV2); // Calculate alpha at new midpoint location
//
//       zp += h; // Update current z position
//
//       // Use the midpoint formula to integrate tau
//       // dtau = - alpha * dzp
//       // tau_(n+1) = tau_n + dzp * alpha(zp_n + h/2)
//       // since we have a simple ODE, explicit and implicit techniques are the same.
//       tau += -h * alpha; // Update tau, remembering h is negative
//
//       expNew = exp(-tau); // Update exp
//
//       // Use the formal solution to calculate the emission coming from this cell, assuming the source function
//       // is constant over the cell
//       tot_intensity += point.S_nu * (expOld - expNew);
//
//       expOld = expNew; //replace expOld with updated tau
//   }
//
//   // Update the return values using pointers.
//   *end_tau = tau;
//   *end_I = tot_intensity;
//
// }
//

// v0 is the central velocity of the channel, relative to the disk systemic velocity
// Do the pre-calculations necessary to call integrate_tau
// void trace_pixel(double xprime, double yprime, double v0, struct pars * p, double DeltaVmax, struct grid * g, double * end_tau, double * end_I)
// {
//
//   double rmax = 700.0 * AU;
//
//   // Based on the radius of the midplane crossing, estimate a safe dtmax
//   // Assume that this has a floor of 0.2 AU.
//   double rcyl_min = fabs(xprime) + 0.2 * AU;
//
//   // we want the zprime that corresponds to the midplane crossing
//   double zprime_midplane = -tan(p->incl * deg) * yprime;
//
//   // Calculate the rcyl_mid here
//   // Corresponds to r_cyl_mid, 0.0
//   double t1 = (cos(p->incl * deg) * yprime + sin(p->incl * deg) * zprime_midplane);
//   double rcyl_mid = sqrt(xprime*xprime + t1 * t1);
//
//   // Get amplitude at midplane crossing
//   // Get scale height at rcyl_min (will be an underestimate)
//   double H_rcyl_min = Hp_pars(rcyl_min, p);
//   double H_rcyl_mid = Hp_pars(rcyl_mid, p);
//
//   double sigma_Upsilon = 0.5 * (H_rcyl_min + H_rcyl_mid)  / cos(p->incl * deg);
//   double max_ds = -sigma_Upsilon / 2.0;
//
//   // Calculate the bounding positions for emission
//   // If it's two, just trace it.
//   // If it's four, break it up into two integrals.
//   struct zps myZps =  get_bounding_zps(xprime, yprime, p, v0, DeltaVmax, rmax);
//
//   struct geoPrecalc gpre = get_geoPrecalc(xprime, yprime, p);
//
//   if (myZps.merge)
//   {
//     integrate_tau(myZps.z1start, myZps.z2end, v0, p, g, gpre, 0.1, 0.0, 0.0, max_ds, end_tau, end_I);
//   }
//   else
//   {
//     integrate_tau(myZps.z1start, myZps.z1end, v0, p, g, gpre, 0.1, 0.0, 0.0, max_ds, end_tau, end_I);
//     if (*end_tau < TAU_THRESH)
//     {
//       integrate_tau(myZps.z2start, myZps.z2end, v0, p, g, gpre, 0.1, *end_tau, *end_I, max_ds, end_tau, end_I);
//     }
//   }
// }



// Define constants as constant memory.
// __constant__ double or something
// __device__ double


// 1 grid of N blocks, each with M threads
// grid

// GPI cruncher
// Total amount of constant memory:               65536 bytes
// Total amount of shared memory per block:       49152 bytes
// Total number of registers available per block: 65536
// Warp size:                                     32
// 16 multiprocessors
// Maximum number of threads per multiprocessor:  2048
// Maximum number of threads per block:           1024
// Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
// Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

// no dimension of launch blocks may exceed 65,535
// We can have 1024 threads per block, so 65,535 * 1024 = 6.7e7
// So, we should be good for our problem if the image size is less than 1024 x 1024 pixels.
// 512 * 512 * 64 = 1.67e7
// 1024 * 1024 * 64 = 6.71e7

// If it exceeds this, then we're going to have to do multiple pixels per thread.

// image size n_pix
// The giant 3D image cube will be (vel, Y, X)
// And this is just written directly to (global) device memory


// I think it makes sense to have one dimension of the block be the frequency dimension,
// Then we have to break each (Y, X) image into sectors.

// Say we have 512 x 512 pixels in each (Y, X) image.
// Then, to break this into sectors

// If we're doing square blocks, then this can be 32 x 32 pixels
// Or, we could just do a column at a time, of 512 pixels or 1024 pixels.

// pixels per channel = n_pix * n_pix
// if we can have 1024 threads per block, then we


// Really need to think about reducing this to ONLY the variables and functions which need
// to be run on the device

__global__ void tracePixel(double *img, int numElements) // img is the DEVICE global memory
{

    int i_vel = blockIdx.x;
    int i_col = blockIdx.y;
    int i_row = threadIdx.x;

    // gridDim.x = n_vel;
    // gridDim.y = n_pix; // columns
    // blockDim.x = n_pix; // rows

    int index = i_vel * (gridDim.y * blockDim.x) + i_col * gridDim.y + i_row;

    // determine whether the pixel should be traced

    if (index < numElements)
    {
        // img[index] = (double) square(index); // just put the index for now.
        img[index] = dPars.M_star;
    }

}

__constant__ struct pars dPars;

// Main routine on the HOST
int main(void)
{

    // Calculate the appropriate constants
    // init_constants();

    // Calculate the velocities
    // double dvel = (vel_end - vel_start) / (n_vel - 1);

    // Create an array of velocities linearly spaced from vel_start to vel_end
    // img.pVel = malloc(n_vel * sizeof(double));
    // for (int i = 0; i < n_vel; i++)
    // {
      // img.pVel[i] = vel_start + dvel * i;
    // }


    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Create parameters on host as constant memory
    struct pars hPars = {.M_star=1.75, .r_c=45.0, .T_10=115., .q=0.63, .gamma=1.0, .Sigma_c=7.0, .ksi=0.14, .dpc=73.0, .incl=45.0, .PA=0.0, .vel=0.0, .mu_RA=0.0, .mu_DEC=0.0};

    // Copy to constant memory on the device
    cudaMemcpyToSymbol(&hPars, &dPars, sizeof(pars));

    // Determine the size of the image, and create memory to hold it, both on the host and on the device.
    int numElements = NVEL * NPIX * NPIX;
    size_t size = numElements * sizeof(double);

    // HOST image memory allocation
    double *h_img = (double *)malloc(size);

    // Verify that allocations succeeded
    if (h_img == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for the image on the host!\n");
        exit(EXIT_FAILURE);
    }

    // DEVICE image memory allocation
    double *d_img = NULL;
    err = cudaMalloc((void **)&d_img, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for the image on the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = NPIX;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    dim3 numBlocks(NVEL, NPIX);
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    tracePixel<<<numBlocks, threadsPerBlock>>>(d_img, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the resulting image in device memory to the host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    // This call requires the kernels to have finished executing, so it's OK.
    err = cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy image from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Save the results to disk using HDF5 files.
    for (int i = 0; i < numElements; ++i)
    {
        printf("i=%d, h_img[%d]=%f\n", i, i, h_img[i]);
    }

    // Create the HDF5 file, overwrite if exists
    hid_t file_id;

    // We're storing a 1D array as the frequencies
    // hsize_t dims_vel[1] = {NVEL};
    // We're storing a 3D array as the image
    hsize_t dims_img[3] = {NVEL, NPIX, NPIX};

    // Create the file ID
    file_id = H5Fcreate("img.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create the double datasets within the file using the H5 Lite interface
    // H5LTmake_dataset(file_id, "/vels", 1, dims_vel, H5T_NATIVE_DOUBLE, img->pVel);
    // H5LTmake_dataset(file_id, "/mask", 3, dims_img, H5T_NATIVE_CHAR, img->pMask);
    // H5LTmake_dataset(file_id, "/tau", 3, dims_img, H5T_NATIVE_DOUBLE, img->pTau);
    H5LTmake_dataset(file_id, "/img", 3, dims_img, H5T_NATIVE_DOUBLE, h_img);

    // Close up
    H5Fclose (file_id);


    err = cudaFree(d_img);  // Free device global memory

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    free(h_img); // Free host memory

    return 0;
}
