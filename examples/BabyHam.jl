
mutable struct BabyHam
    Ns::Int64
    Ls::Float64
    kmul::Array{Float64,2} # wait until everything is clearer
    dx::Float64
    Vtot::Array{Float64,2}               # total potential
    function BabyHam(Lat, Nunit, dx, Vtot)
        Ls = Nunit * Lat;
        Ns = round(Integer, Ls / dx);

        #
        dx = Ls / Ns;
        gridpos = zeros(Ns,1) # allocating as a 2D Array
        gridpos[:,1] = collect(0:Ns-1).'.*dx; #'
        posstart = 0;
        posidx   = 0;
        kx = zeros(Ns,1);
        kx[:,1] = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Ls;
        kmul = kx.^2/2;

        new(Ns, Ls, kmul, dx, Vtot)
    end

end

# importing the necessary functions to comply with Julia duck typing
import Base.*
import Base.A_mul_B!
import Base.eltype
import Base.size
import Base.issymmetric

function *(H::BabyHam, x::Array{Float64,1})
    # matvec overloading
    # Function  to  overload the application of the Hamiltonian times avector
    y_lap  = lap(H,x);
    y_vtot = Vtot(H,x);

    return y_lap + y_vtot;
end

# TODO: this should be optimized in order to take advantage of BLAS 3 operations
function *(H::BabyHam, X::Array{Float64,2})
    # Function  to  overload the application of the Hamiltonian times a matrix
    # Y = zeros(size(X));
    # A_mul_B!(Y, H, X)
    # return Y
    # new version that acts on the full matrices
    Y_lap  = lap(H,X);
    Y_vtot = Vtot(H,X);

    return Y_lap + Y_vtot;
end


function A_mul_B!(Y::Array{Float64,2}, H::BabyHam, V::Array{Float64,2})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end
end

# optimized version for eigs (it uses sub arrays to bypass the inference step)
function A_mul_B!(Y::SubArray{Float64,1,Array{Float64,1}},
                  H::BabyHam,
                  V::SubArray{Float64,1,Array{Float64,1}})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end

end

function size(H::BabyHam)
    return (H.Ns, H.Ns)
end

function eltype(H::BabyHam)
    # we work always in real space, the Fourier operations are on ly meant as
    # a pseudo spectral discretization
    return typeof(1.0)
end

function issymmetric(H::BabyHam)
    return true
end

function lap(H::BabyHam,x::Array{Float64,1})
    # we ask for a vector, given that we will consider the vector to be
    # a nx1 matrix
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = H.kmul.*fft(x);
    return real(ifft(ytemp))
end

function lap(H::BabyHam,x::Array{Float64,2})
    # application of the laplacian part to a matrix.
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication

    ytemp = broadcast(*,H.kmul,fft(x,1));
    return real(ifft(ytemp,1))
end

function Vtot(H::BabyHam,x::Array{Float64,1})
    # application of the potential part of the Hamiltonian to a vector
    return (H.Vtot).*x
end

function Vtot(H::BabyHam, X::Array{Float64,2})
    # application of the potential part of the Hamiltonian to a matric
    return broadcast(*,H.Vtot, X)
end

function compute_psi(H::BabyHam, Neigs, eigstol=1e-12, eigsiter=10000 )
    # we need to add some options to the update
    # functio to solve the eigenvalue problem for a given rho and Vtot

    (ev,psi,nconv,niter,nmult,resid) = eigs(H,   # Hamiltonian
                                                nev=Neigs+3, # number of eigs
                                                which=:SR, # small real part
                                                ritzvec=true, # provide Ritz v
                                                tol=eigstol, # tolerance
                                                maxiter=eigsiter) #maxiter
    # sorting the eigenvalues, eigs already providesd them within a vector
    ind = sortperm(ev);
    # return the eigenvectors
    return psi[:, ind[1:Neigs]];
end

function compute_rho(H::BabyHam, psi)
    return  sum(psi.^2,2)/H.dx;
end

