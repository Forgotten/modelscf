# Scrip to compute the Kohn-Sham map
# input is the total Potential and the output is the charge density
# the input V is the total potential which in this case is just the sum of
# two gaussian bumps of the same height.
# We build the Hamiltonian (baby version), which consists on
# H[V] = -1/2 Lap - V
# we compute the eigenvalue problem H[V]\Psi = \Psi \Lambda,
# where we are only interested in the first Ne eigenvectors
# finaly we compute the charge density

# We point out that we don't consider the chemical potential in this case
#
# 19/04/2018: Leonardo Zepeda-Nunez

include("BabyHam.jl")
using HDF5
FFTW.set_num_threads(round(Integer,Sys.CPU_CORES/2))

 # number of electrons (or in this case Gaussian bumps)
if length(ARGS) > 0
    Ne = parse(Int64,ARGS[1])
else
    Ne = 8
end
# in this case we suppose a simple
Nsamples = 20000;

dx = 0.25
Nunit = 8;
Lat = 10;

Ls = Nunit * Lat;
Ns = round(Integer, Ls / dx);
#
dx = Ls / Ns;
gridpos = zeros(Ns,1) # allocating as a 2D Array
gridpos[:,1] = collect(0:Ns-1).'.*dx;

# generating periodic grid
gridposPer = zeros(3*Ns,1) # allocating as a 2D Array
gridposPer[:,1] = collect(-Ns:2*Ns-1).'.*dx;


sigmaMax = 4.0
sigmaMin = 4.0
coeffMin = 0.8   # min value for the Gaussian depth
coeffMax = 1.2   # max value for the Gaussian depth

# building the Hamiltonian( we will use it extensively)
H = BabyHam(Lat, Nunit, dx, gridpos); # we use a dummy potential in this case

Input = zeros(Ns, Nsamples)
Output = zeros(Ns, Nsamples)

# testing the

for ii = 1:Nsamples

    # we don't want the potentials too close to the boundary of
    # the computational domain
    R = (Lat*Nunit)*rand(1,Ne) ;
    coeff = coeffMin + (coeffMax-coeffMin)*rand(1,Ne);
    sigma = sigmaMin + (sigmaMax-sigmaMin)*rand(1,Ne);

    # we make sure that the potential wells are not to close to each other
    while (minimum(diff(sort(vcat(R[:], R[:]-Lat*Nunit, R[:]+Lat*Nunit))))< 2*sigmaMin )
        R = (Lat*Nunit)*rand(1,Ne);
    end

    println(ii)
    grispos = broadcast(-, gridposPer, R)
    grispos = broadcast(/, grispos, sigma)
    V = -exp.(-grispos.^2/2 )
    V = reshape(sum(broadcast(*, V, coeff), 2), Ns,3)
    V = sum(V,2)

    H.Vtot = V;

    Psi = compute_psi(H, Ne);
    rho = compute_rho(H, Psi[1]);
    println(Psi[2][end]-Psi[2][end-1])

    Input[:,ii] = V[:];
    Output[:,ii] = rho[:];

end



Input_str = string("Input_KS_Ne_", Ne,".h5")
Output_str = string("Output_KS_Ne_", Ne,".h5")

h5write(Input_str, "Input", Input)
h5write(Output_str, "Output", Output)

