function hartree_pot_bc(rho::Array{Float64,1}, Ls::Float64, YukawaK::Float64, epsil0::Float64)
# Calculate the Hartree potential and the energy in periodic boundary potential
# this function is called from the one in the Ham. This is anotated to give the jit
# compiler more information
#
# Lin Lin
# Date of revision: 10/13/2011

    Ns_glb = numel(rho);
    kx = 2*pi* vcat(collect(0:Ns_glb/2),
                    collect(-Ns_glb/2+1:-1))./Ls ;
    # Use Yukawa rather than the bare Coulomb potential.  The method does
    # not depend critically on the fact that the Coulomb kernel is to be
    # used.
    invkx2 = [0;1./(kx[2:end].^2+YukawaK^2)];
    # invkx2 = [1/YukawaK^2;1./(kx(2:end).^2+YukawaK^2)];
    # The 1/YukawaK^2 factor is not important for neutral defect calculation, but
    # might be important for the charged defect calculation.
    Vtemp = 4*pi*invkx2.*fft(rho)  # we can decorate this one
    V  = real(ifft(Vtemp));
 return V / epsil0;
end
