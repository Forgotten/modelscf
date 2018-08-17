# script to check the accuracy of the Neural network electron density
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
using PyPlot

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


Input_str = string("Input_KS_scf_", Natoms,"_sigma_2.0.h5")
Output_str = string("OutputNN_sigma_2.0.h5")
Output_str_ref = string("Output_KS_scf_", Natoms,"_sigma_2.0.h5")

Input =   h5read(Input_str, "Input")
Output =  h5read(Output_str, "Output")
OutputRef =  h5read(Output_str_ref, "Output")

@assert Nsamples == size(Input)[2]

ii = 2

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
if norm(Input[:,ii] - ham.rhoa[:]) > 1.e-8
    print(ii)
end

# total number of occupied orbitals
Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);

# initialize the potentials within the Hemiltonian, setting H[\rho_0]
init_pot!(ham, Nocc)

ham.rho = reshape(Output[:,ii], Ns,1)

# we use the anderson mixing of the potential
mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )

# we use the default options
eigOpts = eigOptions(1.e-12, 1000, "eigs");

scfOpts = scfOptions(1.e-8, 300, eigOpts, mixOpts)


plot(abs.(Output[:,ii] - OutputRef[:,ii]))
println("Error of rho - rho_NN ",
        norm(Output[:,ii] - OutputRef[:,ii]))

(Vnew, err) = update_pot!(ham)
ham.Vtot = Vnew

update_psi!(ham,eigOpts)

PsiNN = deepcopy(ham.psi[:,1:Nocc])
@assert norm(ham.Vtot - Vnew) < 1.e-5

get_force!(ham)
forceNN = ham.atoms.force[:]


# running the scf iteration
@time VtoterrHist = scf!(ham, scfOpts)
println(length(VtoterrHist))
Vref = ham.Vtot

plot(abs.(Vref - Vnew))
println("Error of Vtot given by the Neural Network is ",
        norm(Vref - Vnew) )

get_force!(ham)
forceRef = ham.atoms.force[:]

println("Error in the force ", norm(forceRef - forceNN))

PsiRef = deepcopy(ham.psi[:,1:Nocc])
println("relative Error in the Density Matrix ",
        norm(PsiRef*PsiRef' - PsiNN*PsiNN')/norm(PsiRef*PsiRef'))
#
# ham1 = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)
# if norm(Input[:,ii] - ham1.rhoa[:]) > 1.e-8
#     print(ii)
# end
#
# # total number of occupied orbitals
# Nocc = round(Integer, sum(atoms.nocc) / ham1.nspin);
#
# # initialize the potentials within the Hemiltonian, setting H[\rho_0]
# init_pot!(ham1, Nocc)
#
# ham1.rho = reshape(OutputRef[:,ii], Ns,1)
# mixOpts = andersonMixOptions(ham1.Ns, betamix, mixdim )
# (Vnew1, err) = update_pot!(ham1)
#
# figure(3)
# plot(abs.(Vref - Vnew1))
#
# figure(4)
# plot(abs.(Vnew- Vnew1))
# println("Error of Vtot given by the Neural Network is ",
#         norm(Vnew- Vnew1)/norm(Vnew1))
#
# ########### testing by running on SCF iteration
#
# ham = Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0, Tbeta)
# if norm(Input[:,ii] - ham.rhoa[:]) > 1.e-8
#     print(ii)
# end
#
# # total number of occupied orbitals
# Nocc = round(Integer, sum(atoms.nocc) / ham.nspin);
#
# # initialize the potentials within the Hemiltonian, setting H[\rho_0]
# init_pot!(ham, Nocc)
#
# ham.rho = reshape(Output[:,ii], Ns,1)
#
# # we use the anderson mixing of the potential
# mixOpts = andersonMixOptions(ham.Ns, betamix, mixdim )
#
# plot(abs.(Output[:,ii] - OutputRef[:,ii]))
# println("Error of rho given by the Neural Network is ",
#         norm(Output[:,ii] - OutputRef[:,ii]))
# # updating the potentials afer we loaded the electron density
# update_vtot!(ham, mixOpts)
# update_vtot!(ham, mixOpts)
# println( update_vtot!(ham, mixOpts) )
#
# (Vnew, err) = update_pot!(ham)
#
# @assert norm(ham.Vtot - Vnew) < 1.e-5
#
# update_psi!(ham, eigOpts);
#
# # update the electron density
# update_rho!(ham,Nocc);
#
# # update the total potential, and compute the
# # differnce between the potentials
# Verr = update_vtot!(ham, mixOpts);
