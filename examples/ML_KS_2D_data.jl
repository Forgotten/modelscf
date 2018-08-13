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

include("BabyHam2D.jl")
using HDF5
FFTW.set_num_threads(round(Integer, Sys.CPU_CORES/2))
# in this case we suppose a simple
Nsamples = 20000;

dx = 0.25
Nunit = 2;
Lat = 8;

Ls = Nunit * Lat;
Ns = round(Integer, Ls / dx);
#
dx = Ls / Ns;
gridposX = zeros(Ns,Ns) # allocating as a 2D Array
gridposY = zeros(Ns,Ns)
gridposX = repmat(collect(0:Ns-1).*dx, 1, Ns)
gridposY = repmat(collect(0:Ns-1).'.*dx, Ns,1)

# generating periodic grid
gridposPerX = repmat(collect(-Ns:2*Ns-1).*dx, 1, 3*Ns)
gridposPerY = repmat(collect(-Ns:2*Ns-1).'.*dx, 3*Ns,1)

Ne = 2; # number of electrons (or in this case Gaussian bumps)
sigmaMax = 1
sigmaMin = 1
coeffMin = 0.8   # min value for the Gaussian depth
coeffMax = 1.2   # max value for the Gaussian depth

# building the Hamiltonian( we will use it extensively)
H = BabyHam2D(Lat, Nunit, dx, gridposX); # we use a dummy potential in this case

Input = zeros(Ns,Ns, Nsamples)
Output = zeros(Ns,Ns, Nsamples)
Eigs   = zeros(Ne+1, Nsamples)
# testing the

RxArray = zeros(Ne, Nsamples)
RyArray = zeros(Ne, Nsamples)

for ii = 1:Nsamples

    println(ii)
    # we don't want the potentials too close to the boundary of
    # the computational domain
    Rx = (Lat*Nunit)*rand(1,Ne) ;
    Ry = (Lat*Nunit)*rand(1,Ne) ;

    RxPer = broadcast(+,Rx,Lat*Nunit*repmat([-1; 0; 1],1, 3).'[:])
    RyPer = broadcast(+,Ry,Lat*Nunit*repmat([-1; 0; 1],3, 1)[:])

    coeff = coeffMin + (coeffMax-coeffMin)*rand(1,Ne);
    sigma = sigmaMin + (sigmaMax-sigmaMin)*rand(1,Ne);

    # we make sure that the potential wells are not to close to each other
    @time while min_distance(RxPer[:],RyPer[:]) < 6*sigmaMin
        Rx = (Lat*Nunit)*rand(1,Ne) ;
        Ry = (Lat*Nunit)*rand(1,Ne) ;

        RxPer = broadcast(+,Rx,Lat*Nunit*repmat([-1; 0; 1],1, 3).'[:])
        RyPer = broadcast(+,Ry,Lat*Nunit*repmat([-1; 0; 1],3, 1)[:])
    end

    RxArray[:,ii] = Rx
    RyArray[:,ii] = Ry

    #println(ii)

    V = zeros(size(gridposX));
    for jj = 1 : length(Rx)
        V += create_Gaussian(coeff[jj], sigma[jj],
                             Rx[jj], Ry[jj],
                             gridposPerX,gridposPerY)
    end

    # gridposX = [ gridposPerX-Rx[ii] for ii = 1:length(Rx) ]
    # gridposY = [ gridposPerY-Ry[ii] for ii = 1:length(Ry) ]

    # gridposXY =[ -(gridposX[ii].^2 +gridposY[ii].^2)/(2* sigma[ii]^2)  for ii = 1:length(sigma)]


    # V = [ -coeff[ii]*exp.(gridposXY[ii]) for ii = 1:length(coeff)]
    # # sum in y
    # V = reshape(sum(V), 3*Ns*Ns,3)
    # V = sum(V,2)
    # V = reshape(V,(3*Ns,Ns))
    # # sum in x
    # V = reshape(V.', Ns*Ns,3)
    # V = sum(V,2)
    # V = reshape(V.',(Ns,Ns))


    H.Vtot = V;

    @time (Psi, Eigs_psi) = compute_psi(H, Ne);
    rho = compute_rho(H, Psi);

    Input[:,:,ii] = V;
    Output[:,:,ii] = rho;
    Eigs[:,ii] = Eigs_psi;

end

h5write("Eigs_KS_2D_Ne_2_rand_depth_1.h5", "Eigs", Eigs)
h5write("Input_KS_2D_Ne_2_rand_depth_1.h5", "Input", Input)
h5write("Output_KS_2D_Ne_2_rand_depth_1.h5", "Output", Output)
