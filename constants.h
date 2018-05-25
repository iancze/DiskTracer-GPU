#ifndef CONSTANTS_H
#define CONSTANTS_H 1

// Helpful note of how to organize the constants with extern and header files
// https://stackoverflow.com/questions/1433204/how-do-i-use-extern-to-share-variables-between-source-files


// All of the constants here are declared as `extern` because we will #include constants.h in all of the
// files that we'll want to use these constants.

// Conversion from astronomical units to CGS units
extern double M_sun, M_earth, AU, pc, G, kB, h, c_ang, cc, c_kms;
// Conversion from degrees to radians
extern double deg, nyquist_factor, amu, mu_gas, m_H;

// CO number ratios relative to H nuclei
extern double f_12CO, X_12CO, X_13CO, X_C18O;

// Number ratios relative to average molecule
extern double X_H2, chi_12CO, chi_13CO, chi_C18O, m_CO, m_12CO, m_13CO, m_C18O;

// cutoff tau to stop tracing
extern double TAU_THRESH;

// To hold molecular constants
// Don't need extern here, since this is a structure definition (akin to function?)
struct molecule {
    double X_mol; // number fraction of molecule
    double B0; // rigid rotor constant of molecule (Hz)
    double mu; // (e.s.u. cm) permanent dipole moment of the molecule
    double mol_weight; // (g) molecular weight
    double nu_0; // (Hz) central frequency of transition
    int l; // lower level of transition
    double T_L; // (K) temperature equivalent of energy at lower transition
};

// Structure to contain all molecular constants
extern struct molecule CO12_21, CO12_32, CO13_21, CO13_32, CO18_21, CO18_32;

// function declaration to fill out molecule struct
void init_molecule(struct molecule * p, double X_mol, double B0, double mu, double mol_weight, double nu_0, int l, double T_L);

// function declaration to fill out all constants
void init_constants(void);


// Species can be "12CO", "13CO", etc.
// Transition can be "3-2", "2-1", etc.
// The key to this dictionary is then species * transition

// Instead of a dictionary, this could just be an array, that's specified by the integer from the user.
// Or, just given some central frequency, the transition nearest is chosen automatically.


#endif
