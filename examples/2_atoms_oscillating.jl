# script to obtain the oscillating atoms using a vanilla verlet Integrator
# The main idea is to build a reference model to study the behavior fo the
# Neural network accelerated Molecular dynamic.

# we use the modelscf to compute the forces, and the velocity verlet algorithm
# to evolve the system.

# in this case we save the evolution of the system in a hd5f file.

include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/anderson_mix.jl")
include("../src/kerker_mix.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
include("../src/Integrators.jl")
using HDF5
FFTW.set_num_threads(round(Integer,Sys.CPU_CORES/2))

# getting all the parameters
dx = 0.125;
Nunit = 1;   # number of units
Lat = 8;     # size of the lattice
Ls = Nunit*Lat;
# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 2; # QUESTION: I don't know where this comes from
epsil0 = 10.0;
T_elec = 100.0;

kb = 3.1668e-6;
au2K = 315774.67;
Tbeta = au2K / T_elec;

betamix = 0.5;
mixdim = 10;

Natoms = 2; # number of atoms

sigma  = ones(Natoms,1)*(1.0);  # insulator
omega  = ones(Natoms,1)*0.03;
Eqdist = ones(Natoms,1)*10.0;
mass   = ones(Natoms,1)*42000.0;
nocc   = ones(Natoms,1)*2;          # number of electrons per atom
Z      = nocc;

function forces(x::Array{Float64,1})
    # input
    #       x: vector with the position of the atoms
    # output
    #       f: forces at the center of the atoms

    R = reshape(x, length(x), 1) # we need to make it a two dimensional array
    # creating an atom structure
    atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);
    # allocating a Hamiltonian
    ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)

    # total number of occupied orbitals
    Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

    # setting the options for the scf iteration
    mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )
    eigOpts = eigOptions(1.e-10, 1000, "eig");
    scfOpts = scfOptions(1.e-8, 300, eigOpts, mixOpts)

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    # running the scf iteration
    VtoterrHist = scf!(ham, scfOpts)

    if VtoterrHist[end] > scfOpts.SCFtol
        println("convergence not achieved!! ")
    end

    # we compute the forces
    get_force!(ham)

        # computing the energy
    Vhar = hartree_pot_bc(ham.rho+ham.rhoa,ham);
    # here Vtotnew only considers the

    # NOTE: ham.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    Etot = ham.Eband + 1/2*sum((ham.rhoa-ham.rho).*Vhar)*dx;

    return (ham.atoms.force[:], Etot)
end

# Settign the time evolution

dt = 0.01

x0 = zeros(Natoms); # this is defined as an 2D array
for j = 1:Natoms
  x0[j] = Ls/(Natoms+1) + 2*j
end

v0 = [0.,0.]
x1 = x0 + dt*v0

(x, v, vdot, E) = time_evolution(velocity_verlet, x -> forces(x), dt, 100, x0, x1)

# Pos_str = string("Pos_KS_scf_", Natoms,"_sigma_", sigma[1],".h5")
# isfile(Pos_str) && rm(Pos_str)
# h5write(Pos_str, "R", x)

