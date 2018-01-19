function pseudocharge(gridpos, Ls_glb, atoms, YukawaK,epsil0)
# Create the pseudo-charge for Coulomb interaction

# Lin Lin
# $Date: 3/23/2011. $

Nsglb = length(gridpos);
Natoms = atoms.Natoms;

Z     = atoms.Z;
sigma = atoms.sigma;
R     = atoms.R;

# Calculate the total pseudo-charge
# NOTE: The pseudo-charge should not extend over twice the domain size!
rhoa = zeros(Nsglb,1);
for j=1:Natoms
  d = R[j] - gridpos;
  d = d - round(d/Ls_glb)*Ls_glb;
  # LL: VERY IMPORTANT: Z has minus sign in front of it!
  rhoa = rhoa - Z[j]./sqrt(2*pi*sigma[j]^2) .* (exp(-0.5*(d./sigma[j]).^2 ));
end
end
