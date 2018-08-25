#test construction of the Hamiltonian

include("../src/Atoms.jl")
include("../src/scfOptions.jl")
include("../src/anderson_mix.jl")
include("../src/kerker_mix.jl")
include("../src/Ham.jl")
include("../src/hartree_pot_bc.jl")
include("../src/pseudocharge.jl")
include("../src/getocc.jl")
using HDF5
FFTW.set_num_threads(round(Integer,Sys.CPU_CORES/2))

#number of samples
Nsamples = 200;


dx = 0.5;
Nunit = 16;   # number of units
Lat = 10;     # size of the lattice
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

sigma  = ones(Natoms,1)*(2.0);  # insulator
omega  = ones(Natoms,1)*0.03;
Eqdist = ones(Natoms,1)*10.0;
mass   = ones(Natoms,1)*42000.0;
nocc   = ones(Natoms,1)*2;          # number of electrons per atom
Z      = nocc;

Input = zeros(Ns, Nsamples)
Output = zeros(Ns, Nsamples)


for ii = 1:Nsamples

    R = zeros(Natoms, 1); # this is defined as an 2D array
    # we compute the separation
    ddx = ii*Ls/(2*Nsamples)
    # make sure that the numner of atoms is equals to 2
    @assert Natoms == 2
    R[1] = Ls/2
    R[2] = Ls/2 + 2*sigma[1] + ddx

    # creating an atom structure
    atoms = Atoms(Natoms, R, sigma,  omega,  Eqdist, mass, Z, nocc);

    # allocating a Hamiltonian
    ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)
    Input[:,ii] = ham.rhoa[:];

    # total number of occupied orbitals
    Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

    # initialize the potentials within the Hemiltonian, setting H[\rho_0]
    init_pot!(ham, Nocc)

    # we use the anderson mixing of the potential
    mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )

    # we use the default options
    eigOpts = eigOptions(1.e-12, 1000, "eigs");

    scfOpts = scfOptions(1.e-10, 3000, eigOpts, mixOpts)

    # running the scf iteration
    @time VtoterrHist = scf!(ham, scfOpts)
    Output[:,ii] = ham.Vtot[:];

    println(length(VtoterrHist))
end

Input_str = string("Input_KS_scf_", Natoms,"_sigma_", sigma[1],"_Vtot.h5")
Output_str = string("Output_KS_scf_", Natoms,"_sigma_", sigma[1],"_Vtot.h5")

isfile(Output_str) && rm(Output_str)
isfile(Input_str)  && rm(Input_str)

h5write(Input_str, "Input", Input)
h5write(Output_str, "Output", Output)
