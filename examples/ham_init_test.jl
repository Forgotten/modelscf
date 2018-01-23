#test construction of the Hamiltonian

include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
include("../src/anderson_mix.jl")

dx = 1.0;
Nunit = 32;
Lat = 10;
# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 10; # QUESTION: I don't know where this comes from
epsil0 = 10.0;
T_elec = 100;

kb = 3.1668e-6;
au2K = 315774.67;
Tbeta = au2K / T_elec;


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
# creating an atom structure
atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);

# allocating a Hamiltonian
ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)

# total number of occupied orbitals
Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

# initialize the potentials within the Hemiltonian
# setting H[\rho_0]
init_pot!(ham, Nocc)

# initializing the options
scfOpts = scfOptions();

# running the scf iteration
VtoterrHist = scf!(ham, scfOpts)


# # We define the scfOptions
# scfOpts = scfOptions();
# eigOpts = eigOptions(scfOpts);
# mixOpts = andersonMixOptions(Ns, scfOpts);
# # we test first updating the psi
# tic();
# for ii = 1:scfOpts.scfiter
# update_psi!(ham, eigOpts)
#
# update_rho!(ham,Nocc,Tbeta)
#
# Verr = update_vtot!(ham, mixOpts)
#     if scfOpts.SCFtol > Verr
#         break
#     end
# end
# toc()
