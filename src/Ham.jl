mutable struct Ham
    Ns::Int64
    Ls::Float64
    kmul::Array{Float64,2} # wait until everything is clearer
    dx::Float64
    gridpos::Array{Float64,2}
    posstart
    posidx
    H                  # no idea what is this
    rhoa::Array{Float64,2}               # pseudo-charge for atoms (negative)
    rho                # electron density
    Vhar               # Hartree potential for both electron and nuclei
    Vtot               # total energy
    drhoa  # derivative of the pseudo-charge
    ev
    psi
    fermi
    occ
    nspin
    Neigs::Int64    # QUESTION: Number of eigenvalues?
    atoms
    Eband              # Total band energy
    Fband              # Helmholtz band energy
    Ftot               # Total Helmholtz energy
    YukawaK            # shift for the potential
    epsil0
    Tbeta              # temperature 1beta = 1/T

    function Ham(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0,Tbeta)
        # QUESTION: what is n_extra?
        Ls = Nunit * Lat;
        Ns = round(Integer, Ls / dx);

        #
        dx = Ls / Ns;
        # defining the grid
        gridpos = zeros(Ns,1) # allocating as a 2D Array
        gridpos[:,1] = collect(0:Ns-1).'.*dx; #'
        posstart = 0;
        posidx   = 0;

        # Initialize the atom positions
        atoms   = atoms;
        Neigs = sum(atoms.nocc) + n_extra;

        Ls_glb = Ls;
        Ns_glb = Ns;

        # we define the Fourier multipliers as an 2D array
        kx = zeros(Ns,1);
        kx[:,1] = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Ls;
        kmul = kx.^2/2;

        rhoa = pseudocharge(gridpos, Ls_glb, atoms,YukawaK,epsil0);

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
        nspin = 1;

        new(Ns, Ls, kmul, dx, gridpos, posstart, posidx, H, rhoa,
            rho, Vhar, Vtot, drhoa, ev, psi, fermi, occ,nspin, Neigs, atoms,
            Eband, Fband, Ftot, YukawaK, epsil0, Tbeta)
    end
end

# importing the necessary functions to comply with Julia duck typing
import Base.*
import Base.A_mul_B!
import Base.At_mul_B
import Base.At_mul_B!
import Base.eltype
import Base.size
import Base.issymmetric

function *(H::Ham, x::Array{Float64,1})
    # matvec overloading
    # Function  to  overload the application of the Hamiltonian times avector
    y_lap  = lap(H,x);
    y_vtot = Vtot(H,x);

    return y_lap + y_vtot;
end

# TODO: this should be optimized in order to take advantage of BLAS 3 operations
function *(H::Ham, X::Array{Float64,2})
    # Function  to  overload the application of the Hamiltonian times a matrix
    Y = zeros(size(X));
    A_mul_B!(Y, H, X)
    return Y
end


function A_mul_B!(Y, H::Ham, V)
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end

end

function size(H::Ham)
    return (H.Ns, H.Ns)
end

function eltype(H::Ham)
    # we work always in real space, the Fourier operations are on ly meant as
    # a pseudo spectral discretization
    return typeof(1.0)
end

function issymmetric(H::Ham)
    return true
end


function update_psi!(H::Ham, eigOpts::eigOptions)
    # we need to add some options to the update
    # functio to solve the eigenvalue problem for a given rho and Vtot

    if eigOpts.eigmethod == "eigs"
        # TODO: make sure that eigs works with overloaded operators
        # TODO: take a look a the interface of eigs in Julia
        (ev,psi,nconv,niter,nmult,resid) = eigs(H,   # Hamiltonian
                                                nev=H.Neigs, # number of eigs
                                                which=:SR, # small real part
                                                ritzvec=true, # provide Ritz v
                                                tol=eigOpts.eigstol, # tolerance
                                                maxiter=eigOpts.eigsiter) #maxiter
        #assert(flag == 0);
    end

    # sorting the eigenvalues, eigs already providesd them within a vector
    ind = sortperm(ev);
    # updating the eigenvalues
    H.ev = ev[ind]
    # updating the eigenvectors
    H.psi = psi[:, ind];
end


function update_rho!(H::Ham, nocc::Int64)

    ev = H.ev;
    (occ, fermi) = get_occ(ev, nocc, H.Tbeta);
    occ = occ * H.nspin;
    rho = sum(H.psi.^2*diagm(occ),2)/H.dx;

    # Total energy
    E = sum(ev.*occ);

    # Helmholtz free energy
    intg = H.Tbeta*(fermi-ev);
    ff = zeros(H.Neigs,1);
    for i = 1 : H.Neigs
      if( intg[i] > 30 )  # Avoid numerical problem.
        ff[i] = ev[i]-fermi;
      else
        ff[i] = -1/H.Tbeta * log(1+exp(intg[i]));
      end
    end
    F = sum(ff.*occ) + fermi * nocc * H.nspin;

    H.occ = occ;
    H.fermi = fermi;
    H.Eband = E;
    H.Fband = F;
    H.rho = rho;
end

function lap(H::Ham,x::Array{Float64,1})
    # we ask for a 2 vector, given that we will consider the vector to be
    # a nx1 matrix
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = H.kmul.*fft(x);
    return real(ifft(ytemp))
end

function Vtot(H::Ham,x::Array{Float64,1})
    # application of the potential part of the Hamiltonian
    return (H.Vtot).*x
end



function hartree_pot_bc(rho, H::Ham)
# computes the Hartree potential and energy in periodic boundary potential
# by solving the 1-d Yakuwa equation.
    # we call the (hopefully compiler optmized version)
    return hartree_pot_bc(rho, H.Ls, H.YukawaK, H.epsil0)
end




function update_vtot!(H::Ham, mixOpts)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)
    betamix = mixOpts.betamix;
    mixdim = mixOpts.mixdim;
    ymat = mixOpts.ymat;
    smat = mixOpts.smat;
    iter = mixOpts.iter;

    (Vtotmix,ymat,smat) = anderson_mix(H.Vtot,Vtotnew,
        betamix, ymat, smat, iter, mixdim);

    mixOpts.ymat = ymat;
    mixOpts.smat = smat;
    mixOpts.iter += 1;

    # updating total potential
    H.Vtot = Vtotmix;
    return Verr
end

function init_pot!(H::Ham, nocc::Int64)
    # nocc number of occupied states
    #function to initialize the potential in the Hamiltonian class
    rho  = -H.rhoa;
    rho  = rho / ( sum(rho)*H.dx) * (nocc*H.nspin);
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
    Verr = norm(Vtotnew-H.Vtot)./norm(H.Vtot); # computing the relative error


    # NOTE: H.Fband is only the band energy here.  The real total energy
    # is calculated using the formula below:
    H.Ftot = H.Fband + 1/2 * sum((H.rhoa-H.rho).*H.Vhar)*dx;
    return (Vtotnew,Verr) # returns the differnece betwen two consecutive iterations
end

# TODO: add a default setting for the scfOpts
function scf!(H::Ham, scfOpts::scfOptions)

    # vector containing the hostorical of the results
    VtoterrHist = zeros(scfOpts.scfiter)
    eigOpts = eigOptions(scfOpts);
    mixOpts = andersonMixOptions(H.Ns, scfOpts);

    # number of occupied states
    Nocc = round(Integer, sum(H.atoms.nocc) / ham.nspin);

    # we test first updating the psi

    for ii = 1:scfOpts.scfiter
        # solving the linear eigenvalues problem
        update_psi!(H, eigOpts);

        # update the electron density
        update_rho!(H,Nocc);

        # update the total potential, and compute the
        # differnce between the potentials
        Verr = update_vtot!(H, mixOpts);

        # save the error
        VtoterrHist[ii] = Verr ;
        # test if the problem had already satiesfied the tolerance
        if scfOpts.SCFtol > Verr
            break
        end
    end

    return VtoterrHist[VtoterrHist.>0]
end

function lap_opt(H::Ham,x::Array{Float64,1})
    # we ask for a 2 vector, given that we will consider the vector to be
    xFourier = rfft(x)
    laplacian_fourier_mult!(xFourier, Ls)
    return irfft(xFourier, H.Ns )
end

@inline function laplacian_fourier_mult!(R::Vector{Complex128}, Ls::Float64 )
    c = (2 * pi / Ls)^2
    @inbounds @simd for ii = 1:length(R)
        R[ii] = (ii-1)^2*c*R[ii]
    end
end
