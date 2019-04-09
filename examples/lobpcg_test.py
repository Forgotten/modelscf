# scrip to test the lobpcg_sep eigenvalue solver
include("../src/lobpcg_sep.jl")

using LinearAlgebra

Ns = 100
k = 5 # number of eigenvectors

A = sqrt(Ns)*Diagonal(ones(Ns)) + rand(Ns, Ns)
A = 0.5*(A + A')

(e, X) = eigen(A)

# orthonormal starting guess of the eigenvectors
X0 = qr(rand(Ns, k + 6)).Q[:, 1:k+6]

#computing the lowest K eigenvalues
(eL, XL, it) = lobpcg_sep(A,X0, x-> x, k, verbose=true )

# printing the error
println("error on the computation the eigenvalues " * string(norm(eL - e[1:k])))

# now we use a preconditioner (the exact inverse)
Ainv = inv(A)
(eL1, XL1, it1) = lobpcg_sep(A,X0, x-> Ainv*x, k, verbose=true )

println("error on the computation the eigenvalues " * string(norm(eL1 - e[1:k])))
