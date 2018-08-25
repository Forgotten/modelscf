# script to obtain the oscillating atoms using a vanilla verlet Integrator
# The main idea is to build a reference model to study the behavior fo the 
# Neural network accelerated Molecular dynamic. 

# we use the modelscf to compute the forces, and the velocity verlet algorithm 
# to evolve the system. 

include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/anderson_mix.jl")
include("../src/kerker_mix.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
include("../src/Integrators.jl")


# getting all the parameters
dx = 0.5;
Nunit = 8;   # number of units
Lat = 10;     # size of the lattice
Ls = Nunit*Lat;
# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 10; # QUESTION: I don't know where this comes from
epsil0 = 10.0;
T_elec = 100.0;

kb = 3.1668e-6;
au2K = 315774.67;
Tbeta = au2K / T_elec;

betamix = 0.5;
mixdim = 10;

Ndist  = 1;   # Temporary variable
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
    eigOpts = eigOptions(1.e-8, 1000, "eigs");
    scfOpts = scfOptions(1.e-6, 300, eigOpts, mixOpts)

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    # running the scf iteration
    @time VtoterrHist = scf!(ham, scfOpts)

    if VtoterrHist[end] > scfOpts.SCFtol
        println("convergence not achieved!! ")
    end

    # we compute the forces 
    get_force!(ham)

    return ham.atoms.force[:]
end

# Settign the time evolution

dt = 0.01

x0 = zeros(Natoms); # this is defined as an 2D array
for j = 1:Natoms
  x0[j] = Ls/(Natoms+1) + 2*j
end

x1 = x0 + dt*[1, -1 ]

(x, v, vdot) = time_evolution(velocity_verlet, x -> 10*forces(x), dt, 100, x0, x1)

