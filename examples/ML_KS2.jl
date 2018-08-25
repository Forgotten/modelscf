# Scrip to compute the Kohn-Sham map
# refer to Leonardo Zepeda-Nunez
# test accuracy w.r.t x discretization

include("BabyHam.jl")
using HDF5
FFTW.set_num_threads(4)

# in this case we suppose a simple
Nsamples = 100;

Nunit = 8;
Lat = 10;

Ls = Nunit * Lat;

sigma = 2; # spread of the Gaussian wells
Ne =  2; # number of electrons (or in this case Gaussian bumps)
coeffMin = 0.8   # min value for the Gaussian depth
coeffMax = 1.2   # max value for the Gaussian depth


for ii = 1:Nsamples

    print(ii)
    print('\t')
    # we don't want the potentials too close to the boundary of
    # the computational domain
    R = (Lat*Nunit-4*sigma )*rand(1,Ne) ;
    coeff = coeffMin + (coeffMax-coeffMin)*rand(1,Ne);

    # we make sure that the potential wells are not to close to each other
    while (minimum(diff(sort(R[:])))< 4*sigma )
        R = (Lat*Nunit-4*sigma )*rand(1,Ne);
    end

    dx = 1./4
    Ns = round(Integer, Ls / dx);
    dx = Ls / Ns;
    gridpos = zeros(Ns,1) # allocating as a 2D Array
    gridpos[:,1] = collect(0:Ns-1).'.*dx;

    # generating periodic grid
    gridposPer = zeros(3*Ns,1) # allocating as a 2D Array
    gridposPer[:,1] = collect(-Ns:2*Ns-1).'.*dx;

    # building the Hamiltonian( we will use it extensively)
    H = BabyHam(Lat, Nunit, dx, gridpos); # we use a dummy potential in this case

    V = -exp.(-(broadcast(-, gridposPer, R)/sigma).^2/2 )
    V = reshape(sum( broadcast(*, V, coeff), 2), Ns,3)
    V = sum(V,2)

    H.Vtot = V;

    Psi = compute_psi(H, Ne);
    rho = compute_rho(H, Psi);

    Vcoarse = V[:];
    rhocoarse = rho[:];

    dx = dx / 2
    Ns = round(Integer, Ls / dx);
    dx = Ls / Ns;
    gridpos = zeros(Ns,1) # allocating as a 2D Array
    gridpos[:,1] = collect(0:Ns-1).'.*dx;

    # generating periodic grid
    gridposPer = zeros(3*Ns,1) # allocating as a 2D Array
    gridposPer[:,1] = collect(-Ns:2*Ns-1).'.*dx;

    # building the Hamiltonian( we will use it extensively)
    H = BabyHam(Lat, Nunit, dx, gridpos); # we use a dummy potential in this case

    V = -exp.(-(broadcast(-, gridposPer, R)/sigma).^2/2 )
    V = reshape(sum( broadcast(*, V, coeff), 2), Ns,3)
    V = sum(V,2)

    H.Vtot = V;

    Psi = compute_psi(H, Ne);
    rho = compute_rho(H, Psi);

    Vfine = V[:];
    rhofine = rho[:];


    # check the error
    nn = round(Integer, Ns/2);
    Vf2 = zeros(nn,1)
    rhof2 = zeros(nn,1)
    for k = 1:nn
        Vf2[k,1] = Vfine[2*k-1,1]
        rhof2[k,1] = rhofine[2*k-1,1]
    end
#    Vf2[1,1] = (Vfine[2*nn,1] + 2*Vfine[1,1] + Vfine[2,1]) / 4
#    for k = 2:nn
#        Vf2[k,1] = (Vfine[2*k-2,1]+2*Vfine[2*k-1,1]+Vfine[2*k,1])/4
#    end
#    rhof2[1,1] = (rhofine[2*nn,1] + 2*rhofine[1,1] + rhofine[2,1]) / 4
#    for k = 2:nn
#        rhof2[k,1] = (rhofine[2*k-2,1]+2*rhofine[2*k-1,1]+rhofine[2*k,1])/4
#    end
    print(norm(Vf2 - Vcoarse) / norm(Vf2))
    print('\t')
    println(norm(rhof2 - rhocoarse) / norm(rhof2))
end
