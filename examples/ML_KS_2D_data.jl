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
Nunit = 8;
Lat = 10;

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

Ne = 4; # number of electrons (or in this case Gaussian bumps)
sigmaMax = 2.4
sigmaMin = 1.6
coeffMin = 0.8   # min value for the Gaussian depth
coeffMax = 1.2   # max value for the Gaussian depth

# building the Hamiltonian( we will use it extensively)
H = BabyHam2D(Lat, Nunit, dx, gridposX); # we use a dummy potential in this case

Input = zeros(Ns,Ns, Nsamples)
Output = zeros(Ns,Ns, Nsamples)

# testing the

for ii = 1:Nsamples

    # we don't want the potentials too close to the boundary of
    # the computational domain
    Rx = (Lat*Nunit)*rand(1,Ne) ;
    Ry = (Lat*Nunit)*rand(1,Ne) ;
    coeff = coeffMin + (coeffMax-coeffMin)*rand(1,Ne);
    sigma = sigmaMin + (sigmaMax-sigmaMin)*rand(1,Ne);

    # we make sure that the potential wells are not to close to each other
    while ((minimum(diff(sort(Rx[:])))+minimum(diff(sort(Ry[:]))) )< 5*sigmaMin )
        Rx = (Lat*Nunit)*rand(1,Ne) ;
        Ry = (Lat*Nunit)*rand(1,Ne) ;
    end

    #println(ii)

    V = zeros(size(gridposX))
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

    Psi = compute_psi(H, Ne);
    rho = compute_rho(H, Psi);

    Input[:,:,ii] = V;
    Output[:,:,ii] = rho;

end


h5write("Input_KS_2D_Ne_4_rand_width.h5", "Input", Input)
h5write("Output_KS_2D_Ne_4_rand_width.h5", "Output", Output)
