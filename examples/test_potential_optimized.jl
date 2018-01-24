#test construction of the Hamiltonian
# adding alll the necessary files
include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
include("../src/anderson_mix.jl")

dx = 1.0;
Nunit = 128;
Lat = 10.0;

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
nocc   = ones(Natoms,1)*2;          # number of electrons per atom
Z      = nocc;
atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);

# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 10; # QUESTION: I don't know where this comes from
epsil0 = 10.0;

Ls = Nunit * Lat;
Ns = round(Integer, Ls / dx);

T_elec = 100;

kb = 3.1668e-6;
au2K = 315774.67;
Tbeta = au2K / T_elec;


#
dx = Ls / Ns;
# defining the grid
# allocating gripos as an 2D Array
gridpos = zeros(Ns,1)
gridpos[:,1] = collect(0:Ns-1).'.*dx; #'


ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)

# total number of occupied orbitals
Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

# initialize the potentials within the Hemiltonian
# setting H[\rho_0]
init_pot!(ham, Nocc);

rho = ham.rho;
rhovec = rho[:];

# trigger compilation
pot1 =  hartree_pot_bc(rho, Ls, YukawaK, epsil0);
pot2 = hartree_pot_bc_opt(rho, Ls, YukawaK, epsil0);
pot3 = hartree_pot_bc_opt_vec(rhovec, Ls, YukawaK, epsil0);

# testing the time and the number of allocations
@time pot1 =  hartree_pot_bc(rho, Ls, YukawaK, epsil0);
@time pot2 = hartree_pot_bc_opt(rho, Ls, YukawaK, epsil0);
@time pot3 = hartree_pot_bc_opt_vec(rhovec, Ls, YukawaK, epsil0);

norm(pot1 - pot2 )
