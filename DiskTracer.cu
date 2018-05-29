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
#define NPIX 256
// Number of velocities
#define NVEL 8

// Maximum extent of the image (in AU)
#define RMAX_image 800.0

// number of bins of the texture grid for DeltaV2, S_nu, and Upsilon_nu
#define NR 512
#define NZ 512

// maximum extent (in AU) of the texture grid for DeltaV2, S_nu, and Upsilon_nu
#define RMAX_interp 800.0
#define ZMAX_interp 500.0

// Radius Column for when things start
#define START_COLUMN 16

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

// Function definition for how we'll interpolate using the CUDA texture
__device__ float4 interp_grid(double r, double z);


__managed__ double M_sun = 1.99e33; // [g]
__managed__ double AU = 1.4959787066e13; // [cm]
double pc = 3.0856776e18; // [cm]
__managed__ double G = 6.67259e-8; // [cm3 g-1 s-2]
__managed__ double kB = 1.380658e-16; // [erg K^-1] Boltzmann constant
double h = 6.6260755e-27; //erg s Planck constant
double cc = 2.99792458e10; // [cm s^-1]

// Used when determining the necessary number of pixels in an image, given distance. Anything below
// 2 is not Nyquist sampled. This is currently set to 2.2 to provide a degree of oversampling.
double nyquist_factor = 2.2;

// Atomic
double amu = 1.6605402e-24; // [g]

// Molecular
__managed__ double mu_gas = 2.37; // mean molecular weight of circumstellar gas
__managed__ double m_H = 1.6733e-24; // g mass of hydrogen atom

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
__device__ double TAU_THRESH = 8.0;

// function to initialize molecule structures
// Takes in a pointer to the molecule structure.
void init_molecule(struct molecule * m, double X_mol, double B0, double mu, double mol_weight, double nu_0, int l, double T_L)
{
  // Access the fields of the structure
  // -> is shorthand for (*p).X_mol, etc..
  m->X_mol = X_mol;
  m->B0 = B0;
  m->mu = mu;
  m->mol_weight = mol_weight;
  m->nu_0 = nu_0;
  m->l = l;
  m->T_L = T_L;
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

// ****************************************
// * model functions
// ****************************************

// Assume all inputs to these functions are in CGS units and in *cylindrical* coordinates.

// Calculate temperature in cylindrical coordinates.
__device__ __host__ double temperature(double r, double T_10, double q)
{
  return T_10 * pow(r / (10. * AU), -q);
}

__device__ __host__ double temperature_pars(double r, struct pars * p)
{
  return temperature(r, p->T_10, p->q);
}

// Scale height, calculate in cylindrical coordinates
__device__ __host__ double Hp(double r, double M_star, double T_10, double q) // inputs in cgs
{
  double temp = temperature(r, T_10, q); // [K]
  return sqrt(kB * temp * r*r*r /(mu_gas * m_H * G * M_star)); // [cm]
}

__device__ __host__ double Hp_pars(double r, struct pars * p)
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

// Ksi is microturbulent broadining width in units of km/s.
// Output of this function is in cm/s (e.g., see RADMC manual, eqn 7.12)
double microturbulence(double ksi)
{
  return ksi * 1.e5; // convert from km/s to cm/s
}

// Calculate the partition function for the temperature using Mangum and Shirley expansion.
// Assumes B0 in Hz.
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

// ****************************************
// * geometry functions
// ****************************************


// Precalculate the quantities necessary to make repeated calls to `get_r_cyl`, `get_z`, and `get_vlos` efficient.
__device__ struct geoPrecalc get_geoPrecalc(double xprime, double yprime, struct pars * p)
{

  // Initialize an empty struct
  struct geoPrecalc temp = {};

  // For get_r_cyl
  temp.xp2 = xprime * xprime;
  temp.a1 = cos(p->incl * M_PI/180.) * yprime;
  temp.a2 = sin(p->incl * M_PI/180.);
  // r_cyl = sqrt(xp2 + (a1 - a2 * zprime)^2)

  // For get_z
  temp.b1 = sin(p->incl * M_PI/180.) * yprime;
  temp.b2 = cos(p->incl * M_PI/180.);
  // z = b1 + b2 * zprime

  // For get_vlos
  temp.c1 = sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.) * xprime;
  temp.c2 = xprime * xprime + yprime * yprime;
  // vlos = c1 /(c2 + zprime^2)^(3/4.)

  return temp;
}

// Take a cartesian point in the sky plane and get the cylindrical radius of the disk
// Useful for querying the disk structure or velocity field.
// assumes i is in degrees
__device__ double get_r_cyl(double xprime, double yprime, double zprime, double i)
{
  double temp = (cos(i * M_PI/180.) * yprime - sin(i * M_PI/180.) * zprime);
  return sqrt(xprime * xprime + temp*temp);
}

// Take a cartesian point in the sky plane and get the cylindrical z point in the disk
// assumes i is in degrees
__device__ double get_z(double xprime, double yprime, double zprime, double i)
{
  return sin(i * M_PI/180.) * yprime + cos(i * M_PI/180.) * zprime;
}

// Take a cartesian point in the sky plane and get the line of sight velocity.
// Negative velocity implies a blueshift (towards the observer).
__device__ double get_vlos(double xprime, double yprime, double zprime, struct pars * p)
{
  // According to NVIDIA best practices guide, pg. 50, k^(3/4) can be rewritten as r = sqrt(k); r = r * sqrt(r)
  double temp, k = xprime*xprime + yprime*yprime + zprime*zprime;
  temp = sqrt(k);
  temp = temp * sqrt(temp);
  // temp = (xprime*xprime + yprime*yprime + zprime*zprime)^(3/4.)
  return sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.) * xprime / temp;
}

// get_coords is a function to return the necessary rcyl, z, vlos quickly along a given ray.
__device__ struct coords get_coords(double zprime, struct geoPrecalc gcalc)
{
  // Empty temp struct
  struct coords temp = {};

  double t_rcyl = (gcalc.a1 - gcalc.a2 * zprime);
  temp.rcyl = sqrt(gcalc.xp2 + t_rcyl*t_rcyl);

  temp.z = gcalc.b1 + gcalc.b2 * zprime;

  //vlos = gcalc.c1 / (gcalc.c2 + zprime^2)^(3./4);
  // k^(3/4) can be rewritten as r = sqrt(k); r = r * sqrt(r)
  double t_vlos =  sqrt(gcalc.c2 + zprime*zprime);
  t_vlos = t_vlos * sqrt(t_vlos);
  temp.vlos = gcalc.c1 / t_vlos;

  return temp;
}

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
// Returns a bool true/false
__device__ bool verify_pixel(double xprime, double yprime, struct pars * p, double v0, double DeltaVmax)
{
  double vb_min = (v0 - 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s
  double vb_max = (v0 + 3.0 * DeltaVmax) * 1.0e5; // convert from km/s to cm/s

  double rho2 = xprime*xprime + yprime*yprime;

  if (rho2 > (RMAX_interp*RMAX_interp * AU*AU)) return false;

  bool overlap = (vb_min < 0.0) & (vb_max > 0.0);

  if (xprime > 0.0) {
      if (vb_max < 0.0)
        return false;
      else if (overlap)
        return true;
      else {
        // Calculate (xprime * sqrt(G * pars.M_star * M_sun) * sin(p.incl * deg) / vb_min)^(4/3)
        double temp = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.) / vb_min;
        // k^(4/3) can be written as r = x * cbrt(x)
        return rho2 <= (temp * cbrt(temp));
      }
  }
  else if (xprime < 0.0) {
      if (vb_min > 0.0)
        return false;
      else if (overlap)
        return true;
      else {
        // Calculate (xprime * sqrt(G * pars.M_star * M_sun) * sin(p.incl * deg) / vb_max)^(4/3)
        double temp = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.) / vb_max;
        // k^(4/3) can be written as r = x * cbrt(x)
        return rho2 <= (temp * cbrt(temp));
      }
  }
  else return true;
}


// Helper function to better calculate the 4/3 power (four thirds power => ftp) and reduce clutter
__device__ double ftp(double t, double v)
{
  // (t/v)^(4/3.)
  // according to NVIDIA developers guide, this is best carried out by
  // r = x * cbrt(x);
  double temp = t/v;
  return temp * cbrt(temp);
}


// Get the starting and ending bounding regions on zprime, based only upon the kinematic/geometrical constraints.
// Assumes xprime, yprime, and rmax are in cm. The velocity equivalent to the frequency to be traced is given by v0
// [km/s]. DeltaVmax is the maximum velocity expected along the ray, also in km/s.
__device__ struct zps get_bounding_zps(double xprime, double yprime, struct pars * p, double v0, double DeltaVmax)
{

  // Initialize all to 0
  struct zps myZps = {};

  // The three-sigma bounds on the line profile
  double vb_min = (v0 - 3.0 * DeltaVmax) * 1.e5; // convert from km/s to cm/s
  double vb_max = (v0 + 3.0 * DeltaVmax) * 1.e5; // convert from km/s to cm/s

  // Basically, we can't be larger or smaller than this.
  double v_temp = xprime*xprime + yprime*yprime;
  // to find x^(3/4) do r = sqrt(x); r = r * sqrt(r)
  v_temp = sqrt(v_temp);
  v_temp = v_temp * sqrt(v_temp);
  // v_temp = (xprime^2 + yprime^2)^(3./4)
  double vb_crit = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.)/v_temp;

  // We want to assert that this pixel has already fulfilled the zeroth order check that there will be emission
  // @assert (((xprime >= 0.0) & (vb_max >= 0.0)) | ((xprime <= 0.0) & (vb_min <= 0.0))) "Pixel $xprime, $yprime, will have no emission."

  // There exists a vb=0 velocity, so the best we can say is that z1start and z2end starts and end at (rmax, -rmax),
  // respectively
  bool overlap = (vb_min <= 0.0) && (vb_max >= 0.0);

  // The velocity where the ray intersects the plane of the sky (z^\prime = 0) exists between vb_min and vb_max
  // This means that the two separate bounding regions merge into one.
  bool overlap_crit = (vb_crit > vb_min) && (vb_crit < vb_max);

  double xxyy = xprime*xprime + yprime*yprime;
  double t1 = xprime * sqrt(G * p->M_star * M_sun) * sin(p->incl * M_PI/180.);

  if ((xprime >= 0.0) & (vb_max >= 0.0))
  {
    if (overlap)
    {
      myZps.z1start = RMAX_interp * AU;
      myZps.z2end = -RMAX_interp * AU;
    }
    else if (vb_min >= 0.0)
    {
      myZps.z1start = sqrt(ftp(t1, vb_min) - xxyy);
      myZps.z2end = -sqrt(ftp(t1, vb_min) - xxyy);
    }
    if (overlap_crit)
    {
      // There exists a vb within the range of vbs which yields zprime = 0, so the two regions merge.
      myZps.z1end = 0.0;
      myZps.z2start = 0.0;
      myZps.merge = true;
      return myZps;
    }
    else
    {
      myZps.z1end = sqrt(ftp(t1, vb_max) - xxyy);
      myZps.z2start = -sqrt(ftp(t1, vb_max) - xxyy);
    }
  }
  else if ((xprime <= 0.0) & (vb_min <= 0.0))
  {
    if (overlap)
    {
      myZps.z1start = RMAX_interp * AU;
      myZps.z2end = -RMAX_interp * AU;
    }
    else if (vb_max <= 0.0)
    {
      myZps.z1start = sqrt(ftp(t1, vb_max) - xxyy);
      myZps.z2end = -sqrt(ftp(t1, vb_max) - xxyy);
    }

    if (overlap_crit)
    {
      // There exists a vb within the range of vbs which yields zprime = 0, so the two regions merge.
      myZps.z1end = 0.0;
      myZps.z2start = 0.0;
      myZps.merge = true;
      return myZps;
    }
    else
    {
      myZps.z1end = sqrt(ftp(t1, vb_min) - xxyy);
      myZps.z2start = -sqrt(ftp(t1, vb_min) - xxyy);
    }
  }
  // Return all 4, initialized.
  myZps.merge = false;
  return myZps;
}


// ****************************************
// * Ray-tracing functions
// ****************************************

// Adaptive stepper using Midpoint Method. https://en.wikipedia.org/wiki/Midpoint_method
__device__ void integrate_tau(double zstart, double zend, double v0, struct pars * p, struct geoPrecalc gpre, double h_tau, double tau_start, double intensity_start, double max_ds, double * end_tau, double * end_I)
{

  // @assert max_ds < 0 "max_ds must be a negative number, since the ray is traced along -z"

  double tot_intensity = intensity_start;
  double zp = zstart;
  double tau = tau_start;

  // Write in coordinate conversions for querying alpha and sfunc
  struct coords myCoords = get_coords(zp, gpre);

  // Calculate Delta v at this position
  double Deltav = v0 * 1.e5 - myCoords.vlos;

  // Look up RT quantities from nearest-neighbor interp.
  float4 point = interp_grid(myCoords.rcyl, myCoords.z);

  // Evaluate alpha at current zstart position
  // point.x = DeltaV2
  // point.z = Upsilon
  double alpha = point.z * exp(-Deltav*Deltav/point.x);

  // Midpoint
  double zp2 = 0.0;
  double h;

  double expOld = exp(-tau);
  double expNew = exp(-tau);

  while ((tau < TAU_THRESH) && (zp > zend))
  {
      // Based upon current alpha value, calculate how much of a dz we would need to get dz * alpha = h_tau
      // Because these are negative numbers, the smaller step is actually the maximum (closer to 0)
      h = fmax(-h_tau / alpha, max_ds);

      // Midpoint step position
      zp2 = zp + 0.5 * h;

      // Get the necessary coordinates for querying the grid.
      myCoords = get_coords(zp2, gpre);

      // Calculate Delta v at this position
      Deltav = v0 * 1.e5 - myCoords.vlos;

      // Look up RT quantities from nearest-neighbor interp, evaluated at the midpoint
      point = interp_grid(myCoords.rcyl, myCoords.z);

      // point.x = DeltaV2
      // point.y = S_nu
      // point.z = Upsilon_nu
      alpha = point.z * exp(-Deltav*Deltav/point.x); // Calculate alpha at new midpoint location

      zp += h; // Update current z position

      // Use the midpoint formula to integrate tau
      // dtau = - alpha * dzp
      // tau_(n+1) = tau_n + dzp * alpha(zp_n + h/2)
      // since we have a simple ODE, explicit and implicit techniques are the same.
      tau += -h * alpha; // Update tau, remembering h is negative

      expNew = exp(-tau); // Update exp

      // Use the formal solution to calculate the emission coming from this cell, assuming the source function
      // is constant over the cell
      // S_nu = point.y
      tot_intensity += point.y * (expOld - expNew);

      expOld = expNew; //replace expOld with updated tau
  }

  // Update the return values using pointers.
  *end_tau = tau;
  *end_I = tot_intensity;
}


// v0 is the central velocity of the channel, relative to the disk systemic velocity
// Do the pre-calculations necessary to call integrate_tau
__device__ void trace_pixel(double xprime, double yprime, double v0, struct pars * p, double DeltaVmax, double * end_tau, double * end_I)
{

  // Based on the radius of the midplane crossing, estimate a safe dtmax
  // Assume that this has a floor of 0.2 AU.
  double rcyl_min = fabs(xprime) + 0.2 * AU;

  // we want the zprime that corresponds to the midplane crossing
  double zprime_midplane = -tan(p->incl * M_PI/180.) * yprime;

  // Calculate the rcyl_mid here
  // Corresponds to r_cyl_mid, 0.0
  double t1 = (cos(p->incl * M_PI/180.) * yprime + sin(p->incl * M_PI/180.) * zprime_midplane);
  double rcyl_mid = sqrt(xprime*xprime + t1 * t1);

  // Get amplitude at midplane crossing
  // Get scale height at rcyl_min (will be an underestimate)
  double H_rcyl_min = Hp_pars(rcyl_min, p);
  double H_rcyl_mid = Hp_pars(rcyl_mid, p);

  double sigma_Upsilon = 0.5 * (H_rcyl_min + H_rcyl_mid)  / cos(p->incl * M_PI/180.);
  double max_ds = -sigma_Upsilon / 2.0;

  // Calculate the bounding positions for emission
  // If it's two, just trace it.
  // If it's four, break it up into two integrals.
  struct zps myZps =  get_bounding_zps(xprime, yprime, p, v0, DeltaVmax);

  struct geoPrecalc gpre = get_geoPrecalc(xprime, yprime, p);

  if (myZps.merge)
  {
    integrate_tau(myZps.z1start, myZps.z2end, v0, p, gpre, 0.1, 0.0, 0.0, max_ds, end_tau, end_I);
  }
  else
  {
    integrate_tau(myZps.z1start, myZps.z1end, v0, p, gpre, 0.1, 0.0, 0.0, max_ds, end_tau, end_I);
    if (*end_tau < TAU_THRESH)
    {
      integrate_tau(myZps.z2start, myZps.z2end, v0, p, gpre, 0.1, *end_tau, *end_I, max_ds, end_tau, end_I);
    }
  }
}


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

__constant__ struct pars dPars; // to store parameters
__constant__ double dVels[NVEL]; // to store array of velocities

// using the Texture reference API, since the GPI GPU is compute capability 5.2
// float4 is a CUDA vector type which can be accessed by .x, .y, .z, and .w
// it looks like only float4 is permitted in current versions of CUDA, only double2 is available.
// Since these numbers are changing over large ranges, float is probably fine for this specific application.
// .x = DeltaV2; .y = S_nu; .z = Upsilon_nu, .w = 0.0 (junk);
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;

// query the texture using the actual extent of the grid.
__device__ float4 interp_grid(double r, double z)
{

  // otherwise, convert r and z into normalized texture coordinates and query the texture
  double r_norm = r / (RMAX_interp * AU);
  double z_norm = fabs(z / (ZMAX_interp * AU));

  float4 ans;

  // if r and z are outside of the grid, return a float4 with (1.0e5, 0.0, 0.0, 0.0)
  // e.g., zeros for all parameters except DeltaV2, which is just a large value.
  if ((r_norm > 1.0) || (z_norm > 1.0))
  {
    ans = make_float4(1.0e5, 0.0, 0.0, 0.0);
  }
  else
  {
    ans = tex2D(texRef, r_norm, z_norm);
  }

  return ans;
}

__global__ void tracePixel(bool *mask, double *tau, double *img, int numElements) // img is the DEVICE global memory
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
      // calculate xprime and yprime from the image dimensions
      double xprime = 2 * (i_col - (NPIX / 2)) * RMAX_image * AU/ NPIX;
      double yprime = 2 * (i_row - (NPIX / 2)) * RMAX_image * AU / NPIX;
      double v0 = dVels[i_vel];

      // calculate the minimum r_cyl along this ray
      double rcyl_min = fabs(xprime);

      // get the maxmimum DeltaV2 along this ray.
      float4 ans = interp_grid(rcyl_min, 0.0);
      double DeltaVmax = sqrt(ans.x) * 1.0e-5; // km/s

      bool trace = verify_pixel(xprime, yprime, &dPars, v0, DeltaVmax);
      mask[index] = trace;

      if (trace)
      {
        // Call the ray-tracing routine and store the results in the tau and image arrays.
        trace_pixel(xprime, yprime, v0, &dPars, DeltaVmax, &tau[index], &img[index]);
      }
      else
      {
        // The pixel will not be traced, so set tau and intensity to 0.0
        tau[index] = 0.0;
        img[index] = 0.0;
      }
    }

}


// Main routine on the HOST
int main(void)
{

  // Calculate the appropriate constants
  init_constants();

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Create parameters on host as constant memory
  struct pars hPars = {.M_star=1.75, .r_c=45.0, .T_10=115., .q=0.63, .gamma=1.0, .Sigma_c=7.0, .ksi=0.14, .dpc=73.0, .incl=45.0, .PA=0.0, .vel=0.0, .mu_RA=0.0, .mu_DEC=0.0};

  // Copy parameters to constant memory on the device
  err = cudaMemcpyToSymbol(dPars, &hPars, sizeof(pars));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy parmeters to constant memory (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Calculate the velocities
  double hVels[NVEL];
  // actual velocities will be read in from the data array, but NVEL must be known at compile time.
  double vel_start = -2.5;
  double vel_end = 2.5;
  double dvel = (vel_end - vel_start) / (NVEL - 1.0);
  // Create an array of velocities linearly spaced from vel_start to vel_end
  for (int i = 0; i < NVEL; i++)
  {
    hVels[i] = vel_start + dvel * i;
  }
  // copy to device constant memory
  err = cudaMemcpyToSymbol(dVels, hVels, NVEL * sizeof(double));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy velocities to constant memory (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // BEGIN TEXTURE SETUP -------------------------

  // Set up the texture memory for the grid interpolator.
  // the (32,32,32,32) means we are using a CUDA float4 vector type
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

  // allocate the 2D array to form the texture
  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &channelDesc, NR, NZ);

  // Initialize it with memory from the host
  float4 h_data[NR][NZ]; // array of CUDA 4-vectors
  double r_grid, z_grid, DV2, S, U;

  // Initialize like a row-major image. axis 0 = y, axis 1 = x
  for (int j=0; j<NZ; j++)
  {
    // Since there is a singularity at rcyl = 0 for many of the disk parameterizations, we set the first few
    // columns of the texture < START_COLUMN to be equal to whatever the values are at START_COLUMN
    for (int i=0; i<START_COLUMN; i++)
    {
      r_grid = (START_COLUMN + 0.0)/NR * RMAX_interp * AU;
      z_grid = (j + 0.0)/NZ * ZMAX_interp * AU;

      DV2 = DeltaV2(r_grid, z_grid, &hPars, &CO12_21);
      S = S_nu(r_grid, z_grid, CO12_21.nu_0, &hPars);
      U = Upsilon_nu(r_grid, z_grid, CO12_21.nu_0, &hPars, &CO12_21);

      h_data[j][i] = make_float4(DV2, S, U, 0.0);
    }

    for (int i=START_COLUMN; i<NR; i++)
    {
      r_grid = (i + 0.0)/NR * RMAX_interp * AU;
      z_grid = (j + 0.0)/NZ * ZMAX_interp * AU;

      DV2 = DeltaV2(r_grid, z_grid, &hPars, &CO12_21);
      S = S_nu(r_grid, z_grid, CO12_21.nu_0, &hPars);
      U = Upsilon_nu(r_grid, z_grid, CO12_21.nu_0, &hPars, &CO12_21);

      h_data[j][i] = make_float4(DV2, S, U, 0.0);
    }
  }


  // copy the array we just initialized to the device
  // call signature cudaMemcpytoArray(dArray, wOffset, hOffset, source, size, cudaMemcpyHostToDevice)
  cudaMemcpyToArray(cuArray, 0, 0, h_data,  NR * NZ * sizeof(float4), cudaMemcpyHostToDevice);

  // texture reference parameters
  texRef.addressMode[0] = cudaAddressModeClamp; // clamp to (0.0, 1.0 - 1/N). OR cudaAddressModeBorder (0.0 outside)
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode = cudaFilterModeLinear;     // nearest neighbor: cudaFilterModePoint. cudaFilterModeLinear
  texRef.normalized = true; // true

  cudaBindTextureToArray(texRef, cuArray, channelDesc); // Bind the array to the texture reference

  // END TEXTURE SETUP -------------------------

  // Determine the size of the mask, image, and create memory to hold it on both the host and the device.
  int numElements = NVEL * NPIX * NPIX;
  size_t size_mask = numElements * sizeof(bool);
  size_t size_image = numElements * sizeof(double);

  // HOST mask memory allocation
  bool *h_mask = (bool *)malloc(size_mask);

  // Verify that allocations succeeded
  if (h_mask == NULL)
  {
    fprintf(stderr, "Failed to allocate memory for the mask on the host!\n");
    exit(EXIT_FAILURE);
  }

  // DEVICE mask memory allocation
  bool *d_mask = NULL;
  err = cudaMalloc((void **)&d_mask, size_mask);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate memory for the mask on the device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // HOST tau memory allocation
  double *h_tau = (double *)malloc(size_image);

  // Verify that allocations succeeded
  if (h_tau == NULL)
  {
    fprintf(stderr, "Failed to allocate memory for the tau on the host!\n");
    exit(EXIT_FAILURE);
  }

  // DEVICE tau memory allocation
  double *d_tau = NULL;
  err = cudaMalloc((void **)&d_tau, size_image);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate memory for the tau on the device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // HOST image memory allocation
  double *h_img = (double *)malloc(size_image);

  // Verify that allocations succeeded
  if (h_img == NULL)
  {
    fprintf(stderr, "Failed to allocate memory for the image on the host!\n");
    exit(EXIT_FAILURE);
  }

  // DEVICE image memory allocation
  double *d_img = NULL;
  err = cudaMalloc((void **)&d_img, size_image);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate memory for the image on the device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }


  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = NPIX;
  // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  dim3 numBlocks(NVEL, NPIX);
  printf("CUDA kernel launch with a grid %d x %d (%d blocks) of %d threads\n", numBlocks.x, numBlocks.y, numBlocks.x * numBlocks.y, threadsPerBlock);
  printf("numElements %d\n", numElements);
  tracePixel<<<numBlocks, threadsPerBlock>>>(d_mask, d_tau, d_img, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch tracePixel kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the resulting mask in device memory to the host memory.
  // This call requires the kernels to have finished executing, so it's OK.
  err = cudaMemcpy(h_mask, d_mask, size_mask, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy mask from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the resulting tau in device memory to the host memory.
  // This call requires the kernels to have finished executing, so it's OK.
  err = cudaMemcpy(h_tau, d_tau, size_image, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy tau from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the resulting image in device memory to the host memory.
  // This call requires the kernels to have finished executing, so it's OK.
  err = cudaMemcpy(h_img, d_img, size_image, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy image from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Save the results to disk using HDF5 files.
  hid_t file_id;

  // We're storing a 1D array as the velocities
  hsize_t dims_vel[1] = {NVEL};
  // We're storing a 3D array as the image
  hsize_t dims_img[3] = {NVEL, NPIX, NPIX};

  // Create the file ID
  file_id = H5Fcreate("img.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Create the double datasets within the file using the H5 Lite interface
  H5LTmake_dataset(file_id, "/vels", 1, dims_vel, H5T_NATIVE_DOUBLE, hVels);
  H5LTmake_dataset(file_id, "/mask", 3, dims_img, H5T_NATIVE_CHAR, h_mask);
  // H5LTmake_dataset(file_id, "/mask", 3, dims_img, H5T_NATIVE_DOUBLE, h_mask);
  H5LTmake_dataset(file_id, "/tau", 3, dims_img, H5T_NATIVE_DOUBLE, h_tau);
  H5LTmake_dataset(file_id, "/img", 3, dims_img, H5T_NATIVE_DOUBLE, h_img);

  // Close up
  H5Fclose (file_id);

  // Release the texture memory
  cudaFreeArray(cuArray);

  err = cudaFree(d_mask);  // Free device global memory
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device mask (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_tau);  // Free device global memory
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device tau (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_img);  // Free device global memory
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(h_mask); // Free host memory
  free(h_tau);
  free(h_img);

  return 0;
}
