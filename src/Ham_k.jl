## Hamiltonian with k-point sampling
## CAUTION: all k-points are named q in the code, because k is already used
##          should initiate atoms in the unit cell (R<lat)
## Major changes:
## 1. Ns is changed to the number of grid points in the unit cell
## 2. New fields added to Ham_k: kx (kmul removed), qlist (k-points), q (current k-point)
##    Nunit, Lat are also added as fields
## 3. Lap revised to become 0.5*(\nabla + iq)^2, depends on q (k-point)
## 4. Hamiltonian operator is no longer symmetric, but Hermitian
## 5. Eigenvectors are not guaranteed to be real


mutable struct Ham_k
    Ns::Int64
    Ls::Float64
#    kmul::Array{Float64,2} # wait until everything is clearer
    kx::Array{Float64,2}  # (Yu) to take the place of kmul
    dx::Float64
    gridpos::Array{Float64,2}
    posstart::Int64
    posidx::Int64
    H                  # no idea what is this
    rhoa::Array{Float64,2}               # pseudo-charge for atoms (negative)
    rho::Array{Float64,2}                # electron density
    Vhar::Array{Float64,2}               # Hartree potential for both electron and nuclei
    Vtot::Array{Float64,2}               # total energy
    drhoa  # derivative of the pseudo-charge
    ev::Array{Float64,1}
    psi::Array{Complex{Float64},2}
    fermi::Float64
    occ
    nspin::Int64
    Neigs::Int64    # QUESTION: Number of eigenvalues? 
                    # (Yu Neigs here is the number of electrons in each unit cell)
    atoms::Atoms
    Eband::Float64              # Total band energy
    Fband::Float64               # Helmholtz band energy
    Ftot::Float64                # Total Helmholtz energy
    YukawaK::Float64             # shift for the potential
    epsil0::Float64
    Tbeta::Float64               # temperature 1beta = 1/T
    qlist::Array{Float64,2}      # list of k-vectors (called q to distinguish from basis)
    q::Float64                   # the current k-vector (matrix vector multiplication
                                 # depends on this q)
    Nunit::Int64                 # (Yu) temporary
    Lat::Float64

    function Ham_k(Lat, Nunit, n_extra, dx, atoms,YukawaK, epsil0,Tbeta)
        # QUESTION: what is n_extra?
        # (Yu) Take Nunit as number of unit cells, Lat as lattice vec
        Ls = Nunit * Lat;
        Ns = round(Integer, Lat / dx);  # (Yu) number of grid points, all in one unit cell

        #
        dx = Lat / Ns;
        # defining the grid
        gridpos = zeros(Ns,1) # allocating as a 2D Array # (Yu) why is it 2D?
        gridpos[:,1] = collect(0:Ns-1).'.*dx; #'
        posstart = 0;
        posidx   = 0;

        # Initialize the atom positions # (Yu) can only set atom in one unit cell
                                        #  so that it's constrained to be periodic
        atoms   = atoms;
        Neigs = sum(atoms.nocc) + n_extra;  # (Yu) atoms.nocc is a list of how 
                                            # many electrons each atom contains

#        Ls_glb = Ls;
#        Ns_glb = Ns;

        # we define the Fourier multipliers as an 2D array
        kx = zeros(Ns,1);
        kx[:,1] = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Lat;
#        kmul = kx.^2/2;  # (Yu) this won't be used
        
        # k-vectors
        qlist = zeros(Nunit,1);
        qlist[:,1] = vcat( collect(0:Nunit/2-1), collect( -Nunit/2:-1) )* 2 * pi / Ls;
        q = qlist[1]  # just pick arbitrary one

#        rhoa = pseudocharge(gridpos, Ls_glb, atoms,YukawaK,epsil0);
        rhoa = pseudocharge(gridpos, Lat, atoms,YukawaK,epsil0);

        # TODO: we need to figure out the type of each of the fields to properlu
        # initialize them
        H  = []
        rho = zeros(1,1);  #(Yu) why can't we assign enough memory at the beginning?
        Vhar = zeros(1,1);
        Vtot = zeros(1,1);
        drhoa = []
        ev = []
        psi = zeros(1,1);
        fermi = 0.0;
        occ = []
        Eband = 0.0
        Fband = 0.0
        Ftot = 0.0
        nspin = 1;
        
        Nunit = Nunit;
        Lat = Lat;
        
        new(Ns, Ls, kx, dx, gridpos, posstart, posidx, H, rhoa,
            rho, Vhar, Vtot, drhoa, ev, psi, fermi, occ,nspin, Neigs, atoms,
            Eband, Fband, Ftot, YukawaK, epsil0, Tbeta,qlist,q,Nunit,Lat)
    end
end

# importing the necessary functions to comply with Julia duck typing
import Base.*
import Base.A_mul_B!
import Base.eltype
import Base.size
import Base.issymmetric
import Base.ishermitian

function *(H::Ham_k, x::Array{Complex{Float64},1})
    # matvec overloading
    # Function  to  overload the application of the Hamiltonian times avector
    y_lap  = lap(H,x);
    y_vtot = Vtot(H,x);

    return y_lap + y_vtot;
end

# TODO: this should be optimized in order to take advantage of BLAS 3 operations
function *(H::Ham_k, X::Array{Complex{Float64},2})
    # Function  to  overload the application of the Hamiltonian times a matrix
    Y = zeros(size(X));
    A_mul_B!(Y, H, X)
    return Y
    # # new version that acts on the full matrices
    # Y_lap  = lap(H,X);
    # Y_vtot = Vtot(H,X);
    #
    # return Y_lap + Y_vtot;
end


function A_mul_B!(Y::Array{Complex{Float64},2}, H::Ham_k, V::Array{Complex{Float64},2})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end
end

function A_mul_B!(Y::Array{Float64,2}, H::Ham_k, V::Array{Float64,2})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end
end


# optimized version for eigs (it uses sub arrays to bypass the inference step)
function A_mul_B!(Y::SubArray{Complex{Float64},1,Array{Complex{Float64},1}},
                  H::Ham_k,
                  V::SubArray{Complex{Float64},1,Array{Complex{Float64},1}})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end

end

function size(H::Ham_k)
    return (H.Ns, H.Ns)
end

function eltype(H::Ham_k)
    # (Yu) no longer guaranteed to be in real space
    # a pseudo spectral discretization
    return typeof(1.0*im)
end

function issymmetric(H::Ham_k)
#    return true
    return false
end

function ishermitian(H::Ham_k)
    return true
end


function update_psi!(H::Ham_k, eigOpts::eigOptions)
    # (Yu) this is done for each q
    # we need to add some options to the update
    # functio to solve the eigenvalue problem for a given rho and Vtot

    # (Yu) get Nunit*Neigs ev's in total, should be greater than num of electrons
    H.psi = zeros(H.Ns,H.Neigs*H.Nunit)
    H.ev = zeros(H.Neigs*H.Nunit)
    for idxq = 1:H.Nunit
        H.q = H.qlist[idxq]
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
            # TODO: this has a bug somewhere !!!! fix me!!!!
            # (Yu) removing LOBPCG here
    #    elseif  eigOpts.eigmethod == "lobpcg_sep"
    #        # not working... to be fixed
    #        X0 = qr(rand(H.Ns, H.Neigs), thin = true)[1]
    #        prec(x) = inv_lap(H,x)

    #        (ev,psi, iter) = lobpcg_sep(H, X0, prec, H.Neigs,
    #                            tol= eigOpts.eigstol,
    #                            maxiter=eigOpts.eigsiter)
        end

        # updating the eigenvalues
        H.ev[(idxq-1)*H.Neigs+1:idxq*H.Neigs] = real(ev)
        H.psi[:,(idxq-1)*H.Neigs+1:idxq*H.Neigs] = psi
        # updating the eigenvectors
#        H.psi = psi[:, ind];
    end
    # (Yu) rearrange ev and psi
    # but here it's no longer possible to distinguish between k-points
    ind = sortperm(H.ev);
#    println(size(H.ev))
    H.ev = H.ev[ind]
    H.psi = H.psi[:,ind]
end



function update_rho!(H::Ham_k, nocc::Int64)

    ev = H.ev;
    (occ, fermi) = get_occ(ev, nocc, H.Tbeta);
    occ = occ * H.nspin;
#    println(occ)
    # (Yu) cannot expect psi to be real here
    rho = sum((abs.(H.psi).^2)*diagm(occ),2)/H.dx/H.Nunit;  
#    println(sum(rho)*H.dx)

    # Total energy
    E = sum(ev.*occ);

    # Helmholtz free energy
    # (Yu) TODO check whether revision is needed
    intg = H.Tbeta*(fermi-ev);
    ff = zeros(H.Neigs*H.Nunit,1);
    for i = 1 : H.Neigs*H.Nunit
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
#    println(minimum(H.rho))
end


function lap(H::Ham_k,x::Array{Complex{Float64},1})
    # (Yu) revised Laplacian, taking into account the k-point
    # we ask for a vector, given that we will consider the vector to be
    # a nx1 matrix
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = (0.5*(H.kx+H.q).^2).*fft(x);  # take into account the wavevector
#    return real(ifft(ytemp))
    return ifft(ytemp)
end

function Vtot(H::Ham_k,x::Array{Complex{Float64},1})
    # application of the potential part of the Hamiltonian to a vector
    return (H.Vtot).*x
end

function Vtot(H::Ham_k, X::Array{Complex{Float64},2})
    # application of the potential part of the Hamiltonian to a matric
    return broadcast(*,H.Vtot, X)
end


function hartree_pot_bc(rho, H::Ham_k)
# computes the Hartree potential and energy in periodic boundary potential
# by solving the 1-d Yakuwa equation.
    # we call the (hopefully compiler optmized version)
    # (Yu) Ls replaced by Lat
    return hartree_pot_bc(rho, H.Lat, H.YukawaK, H.epsil0)
end




function update_vtot!(H::Ham_k, mixOpts)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)

    # TODO: add a swtich to use different kinds of mixing here
    betamix = mixOpts.betamix;
    mixdim = mixOpts.mixdim;
    ymat = mixOpts.ymat;
    smat = mixOpts.smat;
    iter = mixOpts.iter;

    (Vtotmix,ymat,smat) = anderson_mix(H.Vtot,Vtotnew,
        betamix, ymat, smat, iter, mixdim);

    # they are already modified inside the function
    # mixOpts.ymat = ymat;
    # mixOpts.smat = smat;
    mixOpts.iter += 1;

    # updating total potential
    H.Vtot = Vtotmix;
    return Verr
end

function update_vtot!(H::Ham_k, mixOpts::kerkerMixOptions)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)

    # println("using kerker mixing")
    # computign the residual
    res     = H.Vtot - Vtotnew;

    # println("appliying the preconditioner")
    resprec = kerker_mix(res, mixOpts.KerkerB, mixOpts.kx,
                              mixOpts.YukawaK, mixOpts.epsil0);

    # println("performing the linear mixing ")
    Vtotmix =  H.Vtot - mixOpts.betamix * resprec;

    # updating total potential
    # println("saving the total potentail ")
    H.Vtot = Vtotmix;

    # println("returning the error")
    return Verr
end

function update_vtot!(H::Ham_k, mixOpts::andersonPrecMixOptions)
    # TODO here we need to implement the andersson mix
    # I added the signature

    (Vtotnew,Verr) = update_pot!(H)


    betamix = mixOpts.betamix;
    mixdim = mixOpts.mixdim;
    ymat = mixOpts.ymat;
    smat = mixOpts.smat;
    iter = mixOpts.iter;

    (Vtotmix,ymat,smat) = prec_anderson_mix(H.Vtot,Vtotnew,
        betamix, ymat, smat, iter, mixdim, mixOpts.prec, mixOpts.precargs)

    # they are already modified inside the function
    # mixOpts.ymat = ymat;
    # mixOpts.smat = smat;
    mixOpts.iter += 1;

    # updating total potential
    H.Vtot = Vtotmix;
    return Verr
end

function init_pot!(H::Ham_k, nocc::Int64)
    # nocc number of occupied states
    #function to initialize the potential in the Hamiltonian class
    rho  = -H.rhoa;
    rho  = rho / ( sum(rho)*H.dx) * (nocc*H.nspin);  # (Yu) !MARK! should *H.dx be included here?
    H.rho = rho;
    H.Vhar = hartree_pot_bc(H.rho+H.rhoa, H);
    H.Vtot = H.Vhar;   # No exchange-correlation
    H.Vtot = H.Vtot - mean(H.Vtot);# IMPORTANT (zero mean?)
end

function update_pot!(H::Ham_k)
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
function scf!(H::Ham_k, scfOpts::scfOptions)

    # vector containing the hostorical of the results
    VtoterrHist = zeros(scfOpts.scfiter)
    eigOpts = scfOpts.eigOpts;
    mixOpts = scfOpts.mixOpts;

    # number of occupied states
    Nocc = round(Integer, sum(H.atoms.nocc) / H.nspin)*H.Nunit; 

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

# (Yu) not sure what's going on here, not using them for the moment
#function lap_opt(H::Ham,x::Array{Float64,1})
#    # we ask for a 2 vector, given that we will consider the vector to be
#    xFourier = rfft(x)
#    laplacian_fourier_mult!(xFourier, H.Ls)
#    return irfft(xFourier, H.Ns )
#end

#function lap_opt!(H::Ham,x::Array{Float64,1})
#    # we ask for a 2 vector, given that we will consider the vector to be
#    xFourier = rfft(x)
#    laplacian_fourier_mult!(xFourier, H.Ls)
#    x[:] = irfft(xFourier, H.Ns )
#end

#function laplacian_fourier_mult!(R::Vector{Complex128}, Ls::Float64 )
#    c = (2 * pi / Ls)^2/2
#    @simd for ii = 1:length(R)
#        @inbounds R[ii] = (ii-1)^2*c*R[ii]
#    end
#end
