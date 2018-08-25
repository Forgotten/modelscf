include("BabyHam.jl")


# Scrip to compute the Kohn-Sham map
# input is the total Potential and the output is the charge density

# we need to create a Hamiltonian

# in this case we suppose a simple

dx = 0.25
Nunit = 8;
Lat = 10;

Ls = Nunit * Lat;
Ns = round(Integer, Ls / dx);
#
dx = Ls / Ns;
gridpos = zeros(Ns,1) # allocating as a 2D Array
gridpos[:,1] = collect(0:Ns-1).'.*dx;

sigma = 2; 
Ne =  2; # number of electrons (or in this case Gaussian bumps)

R = [10, 20]

V = sum(-exp.(-(broadcast(-, gridpos, R')/sigma).^2/2 ), 2)


# testing the 

H = BabyHam(Lat, Nunit, dx, V);

H.Vtot = V;

Psi = compute_psi(H, Ne);

rho = compute_rho(H, Psi);