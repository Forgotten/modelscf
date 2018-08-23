function  velocity_verlet{T<:AbstractFloat}(a::Function,
                                              x::Array{T,1},
                                              v::Array{T,1},
                                              vdot::Array{T,1},
                                              h::T)
# modification to have only one evaluation of a per iteration
    v_half = v      + h/2 * vdot
    x_next = x      + h   * v_half
    vdot_next = a(x_next)
    v_next = v_half + h/2 * vdot_next
    return (x_next, v_next, vdot_next)
end

function time_evolution{T<:AbstractFloat}(Integrator::Function,
                                          a::Function,
                                          h::T,
                                          Nsteps::Int64,
                                          x0::Array{T,1},
                                          x1::Array{T,1} )

xArray = Array{T,2}(Nsteps+1, length(x0))
vArray = Array{T,2}(Nsteps+1, length(x0))
aArray = Array{T,2}(Nsteps+1, length(x0))

println("initial conditions")
xArray[1,:] =  x0
xArray[2,:] =  x1

vArray[1,:] = (x1 - x0)/(h) # is this enough?
aArray[1,:] = a(x0)

println("running the loop")

for ii = 1:Nsteps
    (x, v, vdot) = Integrator(a,xArray[ii,:], vArray[ii,:], aArray[ii,:], h)
    xArray[ii+1,:] = x
    vArray[ii+1,:] = v
    aArray[ii+1,:] = vdot
end
println("loop finished")

return (xArray, vArray, aArray)
end



