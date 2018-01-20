struct Ham
    Ns::Int64
    Ls::Float64
    kmul ## ::Array{Float64,2} wait until everything is clearer
    dx::Float64
    gridpos
    posstart
    posidx
    H                  # no idea what is this
    lap
    rhoa               # pseudo-charge for atoms (negative)
    rho                # electron density
    Vhar               # Hartree potential for both electron and nuclei
    Vtot               # total energy
    drhoa              # derivative of the pseudo-charge
    ev
    psi
    fermi
    occ
    Neigs
    atoms
    Eband              # Total band energy
    Fband              # Helmholtz band energy
    Ftot               # Total Helmholtz energy

    function Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0)
        # QUESTION: what is n_extra?
        Ls = Nunit * Lat;
        Ns = round(Int, Ls / dx);

        #
        dx = Ls / Ns;
        # defining the grid
        gridpos = collect(0:Ns-1)'.*dx; #'
        posstart = 0;
        posidx   = 0;

        # Initialize the atom positions
        atoms   = atoms;
        Neigs = sum(atoms.nocc) + n_extra;

        Ls_glb = Ls;
        Ns_glb = Ns;

        kx = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Ls;
        kmul = kx.^2/2;

        rhoa = pseudocharge(gridpos, Ls_glb, atoms,YukawaK,epsil0);

        rhoa  = rhoa;

        # TODO: we need to figure out the type of each of the fields to properlu
        # initialize them
        H  = []
        rho = []
        Vhar = []
        Vtot = []
        drhoa = []
        ev = []
        psi = []
        fermi = []
        occ = []
        Eband = []
        Fband = []
        Ftot = []

        new(Ns, Ls, kmul, gridpos, posstart, posidx, H, lap, rhoa,
            rho, Vhar, Vtot, drhoa, ev, psi, fermi, occ, Neigs, atoms,
            Eband, Fband, Ftot)
    end
end

import Base.*

function *(H::Ham, x::Array{Float64,2})
    # Function  to  overload the application of the Hamiltonian times avector
    y_lap = inv_lap(H,x);
    y_lap = Vtot(H,x);

    return y_lap + y_lap;
end

function updateX(H::Ham)

    ## we really need to take a look at the dependencies in here
    opts.issym  = 1;
    opts.isreal = 1;
    opts.tol    = 1e-8;
    opts.maxit  = 1000;

    # TODO: make sure that eigs works with overloaded operators
    # TODO: take a look a the interface of eigs in Julia
    results = eigs(H, H.Ns, H.Neigs, opts);
    assert(flag == 0);

    ev = diag(ev);

    #sorting the eigenvalues
    ind = sortperm(ev);
    ev = ev[ind]
    psi = psi(:, ind);

    (occ, fermi) = get_occ(ev, nocc, Tbeta);
    occ = occ * nspin;
    rho = sum(psi.^2*diag(occ),2)/dx;

    # Total energy
    E = sum(ev.*occ);

    # Helmholtz free energy
    intg = Tbeta*(fermi-ev);
    ff = zeros(Neigs,1);
    for i = 1 : Neigs
      if( intg(i) > 30 )  # Avoid numerical problem.
        ff(i) = ev(i)-fermi;
      else
        ff(i) = -1/Tbeta * log(1+exp(intg(i)));
      end
    end
    F = sum(ff.*occ) + fermi * nocc * nspin;

    return (F, E, rho, ev, psi, occ, fermi)

end


function update_rho(H::Ham)

end

function inv_lap(H::Ham,x::Array{Float64,2})
    # we ask for a 2 vector, given that we will consider the vector to be
    # a nx1 matrix
    ytemp = H.kmul.*fft(x);
    return real(ifft(ytemp))
end

function Vtot(H::Ham,x::Array{Float64,2})
    # application of the potential part of the Hamiltonian
    return (H.Vtot).*x
end



function hartree_pot_bc(rho, H::Ham)
# computes the Hartree potential and energy in periodic boundary potential
# by solving the 1-d Yakuwa equation.
    # we call the (hopefully compiler optmized version)
    return hartree_pot_bc(rho, H.Ls, H.YukawaK, H.epsil0)
end

function mixing()
    # TODO here
    # here we need to implement the andersson mix
      (Vtotmix,ymat,smat) = andersonmix(glb.Vtot,Vtotnew,
        betamix(1),ymat,smat, iter, mixdim);
end

function init_pot(H::Ham)
    #function to initialize the potential in the Hamiltonian class

    rho  = -H.rhoa;
    rho  = rho / (sum(rho)*H.dx) * (H.nocc*H.nspin);
    H.rho = rho;
    H.Vhar = hartree_pot_bc(H.rho+H.rhoa, H);
    H.Vtot = H.Vhar;   # No exchange-correlation
    H.Vtot = H.Vtot - mean(H.Vtot);# IMPORTANT (zero mean?)
end

function update_pot!(H::Ham)
     # TODO: I dont' know in how many different ways this si updateded
     # computing the hartree potenatial
    H.Vhar = hartree_pot_bc(H.rho+H.rhoa,H);
    # here Vtotnew only considers the
    Vtotnew  = H.Vhar;  # no exchange-correlation so far
    Verr = norm(Vtotnew-H.Vtot)./norm(H.Vtot);


    # NOTE: H.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    H.Ftot = H.Fband + 1/2 * sum((H.rhoa-H.rho).*H.Vhar)*dx;
    return Verr # returns the differnece betwen two consecutive iterations
end
