
mutable struct BabyHam2D
    Ns::Int64
    Ls::Float64
    kmul::Array{Float64,2} # wait until everything is clearer
    dx::Float64
    Vtot::Array{Float64,2}               # total potential
    function BabyHam2D(Lat, Nunit, dx, Vtot)
        Ls = Nunit * Lat;
        Ns = round(Integer, Ls / dx);

        #
        dx = Ls / Ns;

        # checking that the sizes of the potential matches
        @assert size(Vtot)[1] == Ns && size(Vtot)[2] == Ns

        kx = zeros(Ns,1);
        kx[:,1] = vcat( collect(0:Ns/2-1), collect( -Ns/2:-1) )* 2 * pi / Ls;
        Kx = repmat(kx, 1, Ns)
        Ky = repmat(kx.', Ns, 1)
        kmul = (Kx.^2 + Ky.^2)/2;

        new(Ns, Ls, kmul, dx, Vtot)
    end

end

# importing the necessary functions to comply with Julia duck typing
import Base.*
import Base.A_mul_B!
import Base.eltype
import Base.size
import Base.issymmetric

function *(H::BabyHam2D, x::Array{Float64,1})
    # matvec overloading
    # Function  to  overload the application of the Hamiltonian times avector
    @assert length(x) == length(H.kmul)
    y_lap  = lap(H,x);
    y_vtot = Vtot(H,x);

    return y_lap + y_vtot;
end

# TODO: this should be optimized in order to take advantage of BLAS 3 operations
function *(H::BabyHam2D, X::Array{Float64,2})
    # Function  to  overload the application of the Hamiltonian times a matrix
    # Y = zeros(size(X));
    # A_mul_B!(Y, H, X)
    # return Y
    # new version that acts on the full matrices
    Y_lap  = lap(H,X);
    Y_vtot = Vtot(H,X);

    return Y_lap + Y_vtot;
end

function min_distance(X::Array{Float64,1}, Y::Array{Float64,1})
    Xall = broadcast(-,X , X')
    Yall = broadcast(-,Y , Y')

    dist = sqrt.(Xall.^2 + Yall.^2)

    return minimum(sort(dist,2)[:,2])
end

function A_mul_B!(Y::Array{Float64,2}, H::BabyHam2D, V::Array{Float64,2})
    # in place matrix matrix multiplication
    assert(size(Y) == size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end
end

# optimized version for eigs (it uses sub arrays to bypass the inference step)
function A_mul_B!(Y::SubArray{Float64,1,Array{Float64,1}},
                  H::BabyHam2D,
                  V::SubArray{Float64,1,Array{Float64,1}})
    # in place matrix matrix multiplication
    @assert(size(Y) == size(V))
    # print(size(V))
    for ii = 1:size(V,2)
        Y[:,ii] = H*V[:,ii]
    end

end

function size(H::BabyHam2D)
    return (H.Ns.^2, H.Ns.^2)
end

function eltype(H::BabyHam2D)
    # we work always in real space, the Fourier operations are on ly meant as
    # a pseudo spectral discretization
    return typeof(1.0)
end

function issymmetric(H::BabyHam2D)
    return true
end

function lap(H::BabyHam2D,x::Array{Float64,1})
    # we ask for a vector, given that we will consider the vector to be
    # a nx1 matrix
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication
    xM = reshape(x, size(H.kmul))
    y = fft(xM)
    ytemp = H.kmul.*y;
    return real(ifft(ytemp)[:])
end

function lap(H::BabyHam2D,x::Array{Float64,2})
    # application of the laplacian part to a matrix.
    # TODO: this can be optimized using rfft
    # TODO: we can surther accelerate this using a in-place multiplication
    nx,ny = size(H.kmul)
    @assert size(x)[1] == nx*ny
    nSamples = size(x)[2]
    xM = reshape(x,(nx,ny,nSamples))
    ytemp = broadcast(*,H.kmul,fft(xM,(1,2)));

    return real(reshape(ifft(ytemp,(1,2) ),(nx*ny,nSamples) ))
end

function Vtot(H::BabyHam2D,x::Array{Float64,1})
    # application of the potential part of the Hamiltonian to a vector
    return ((H.Vtot).*reshape(x, size(H.kmul)))[:]
end

function Vtot(H::BabyHam2D, X::Array{Float64,2})
    # application of the potential part of the Hamiltonian to a matric
    return broadcast(*,H.Vtot, X)
end

function compute_psi(H::BabyHam2D, Neigs, eigstol=1e-12, eigsiter=10000 )
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
    return (psi[:, ind[1:Neigs]], ev[1:Neigs+1]);
end

function compute_rho(H::BabyHam2D, psi)
    return  reshape(sum(psi.^2,2)/H.dx.^2, (H.Ns, H.Ns));
end


function create_Gaussian(coeff, sigma, Rx, Ry, gridposPerX,gridposPerY)
    gridposX = gridposPerX-Rx
    gridposY = gridposPerY-Ry

    gridposXY = -(gridposX.^2 +gridposY.^2)/(2* sigma^2)

    V = -coeff*exp.(gridposXY)
    # sum in the both direction to mae it truly periodic
    # sum in y
    V = reshape(V, 3*Ns*Ns,3)
    V = sum(V,2)
    V = reshape(V,(3*Ns,Ns))
    # sum in x
    V = reshape(V.', Ns*Ns,3)
    V = sum(V,2)
    V = reshape(V.',(Ns,Ns))

    return V

end
