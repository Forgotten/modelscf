# scriopt to tes that the NN force computation is accurate enough
using PyCall
using PyPlot

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

filename = "NN_MD_SCF.py"

@pyimport imp
(path, name) = dirname(filename), basename(filename)
(name, ext) = rsplit(name, '.')

(file, filename, data) = imp.find_module(name, [path])

NN = imp.load_module(name, file, filename, data)

NNrho = NN[:NN_MD]()
#testing the function properly

NNrho[:eval](rand(64))

dx = 0.125;
Nunit = 1;   # number of units
Lat = 8;     # size of the lattice
Ls = Nunit*Lat;

Ns = round(Integer, Ls / dx);
# using the default values in Lin's code
YukawaK = 0.0100
n_extra = 10; # QUESTION: I don't know where this coAmes from
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

numSCFit = 1

function forcesNN(x::Array{Float64,1})
    # input
    #       x: vector with the position of the atoms
    # outputpl
    #       f: forces at the center of the atoms

    R = reshape(x, length(x), 1) # we need to make it a two dimensional array
    # creating an atom structure
    atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);
    # allocating a Hamiltonian
    ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta);
    
    rho_NN = NNrho[:eval](-ham.rhoa-0.5);

    # total number of occupied orbitals
    Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

    # setting the options for the scf iteration
    mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )
    eigOpts = eigOptions(1.e-10, 1000, "eig");
    scfOpts = scfOptions(1.e-8, numSCFit, eigOpts, mixOpts)

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    ################
    # ham.rho = reshape(rho_NN, Ns,1)

    # (Vnew, err) = update_pot!(ham)
    # ham.Vtot = Vnew

    # update_psi!(ham,eigOpts)

    # update_rho!(ham, Nocc)
    # # we compute the forces
    # get_force!(ham)

    # Vhar = hartree_pot_bc(ham.rho+ham.rhoa,ham);
    # # here Vtotnew only considers the

    # # NOTE: ham.Fband is only the band energy here.  The real total energy
    # # is calculated using the formula below:
    # Etot = ham.Eband + 1/2*sum((ham.rhoa-ham.rho).*Vhar)*dx;
    #################

    ham.rho = reshape(rho_NN, Ns,1)

    (Vnew, err) = update_pot!(ham)

     for ii = 1:scfOpts.scfiter
        # solving the linear eigenvalues problem
        update_psi!(ham, eigOpts);

        # update the electron density
        update_rho!(ham,Nocc);

        # update the total potential, and compute the
        # differnce between the potentials
        Verr = update_vtot!(ham, mixOpts);
        #println(Verr)
    end

    # getting the forces 
    get_force!(ham)

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

v0 = [0.0,0.0]
x1 = x0 + dt*v0

(x, v, vdot, E) = time_evolution(velocity_verlet, x -> forcesNN(x), dt, 30000, x0, x1)

# Pos_str = string("Pos_KS_scf_", Natoms,"_sigma_", sigma[1],"_NN.h5")
# isfile(Pos_str) && rm(Pos_str)
# h5write(Pos_str, "R", x)
