# script to show how to optimize the calls for the Fourier transform
# we gain a factor of 2 roughly for problems big enough

function ddrddx(R::Array{Float64,1}, N::Int64, Ls::Float64 )
l = fft(R)
k = vcat( collect(0:N/2-1), collect( -N/2:-1) )* 2 * pi / Ls;

ddrdxx = real(ifft(k.^2.*l))

return ddrdxx
end


function laplacian_fourier_mult!(R::Array{Complex128,1}, Ls::Float64 )
    c = (2 * pi / Ls)^2
    @simd for ii = 1:length(R)
        @inbounds R[ii] = (ii-1)^2*c*R[ii]
    end
end

function ddrddx2(R::Array{Float64,1}, N::Int64, Ls::Float64 )

    rFourier = rfft(R)
    laplacian_fourier_mult!(rFourier, Ls)

return irfft(rFourier, N )
end


N = 100;
Ls = 1.0;
R = sin.(Ls *2*pi*collect(0:N-1)/N)

@time dRdx = ddrddx(R, N, Ls);
@time dRdx2 = ddrddx2(R, N, Ls);
