#test construction of the Hamiltonian

include("Atoms.jl")
include("Ham.jl")
include("hartree_pot_bc.jl")
include("pseudocharge.jl")


dx = 1.0;
Nunit = 32;
Lat = 1

Ndist  = 1;   # Temporary variable
Natoms = round(Integer, Nunit / Ndist);
# Temp var, will be redefined by the size of R later.
R = zeros(Natoms, 1); # this is defined as an 2D array
for j = 1:Natoms
  R[j] = (j-0.5)*Lat*Ndist+dx;
end

Natoms = size(R)[1];
sigma  = ones(Natoms,1)*(2.0);  # insulator
omega  = ones(Natoms,1)*0.03;
Eqdist = ones(Natoms,1)*10.0;
mass   = ones(Natoms,1)*42000.0;
nocc   = ones(Natoms,1)*2;
Z      = nocc;
atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);

n_extra = 0; # QUESTION: I don't know where this comes from
ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0)
