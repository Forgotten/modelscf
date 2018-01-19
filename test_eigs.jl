struct test
    M::Array{Float64,2}

    function test(M::Array{Float64,2})
        new(M);
    end
end

import Base.*
import Base.A_mul_B!
import Base.At_mul_B
import Base.At_mul_B!
import Base.eltype
import Base.size
import Base.issymmetric


function At_mul_B(T::test, v)
    return T.M'*v
end


# We need be carefull when modifying input variables
function At_mul_B!(y, T::test, v)
    y[:,:]= T.M'*v
end
#
function A_mul_B!(Y, T::test, V)
    Y[:,:] = T.M*V;
end

function size(T::test)
    return size(T.M)
end

function eltype(T::test)
    return eltype(T.M)
end

function issymmetric(T::test)
    return issymmetric(T.M)
end

function *(H::test, x::Array{Float64,2})
# matvec overloading
    return H.M*x
end


n = 10;
M = rand(n,n)+ eye(n);
T = test(M);
r = rand(n,1)
assert(norm(M*r - T*r) < 1e-12)

B = rand(n,n)
Y = zeros(n,n)
At_mul_B!(Y,T,B)

assert(norm(Y - M'*B) < 1e-12)


eigs(T)
