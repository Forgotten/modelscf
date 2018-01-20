struct Atoms
 # atoms struct to store the data from the atoms
     Natoms         #::Int64
     R              #::Array{Float64,1}
     sigma          #::Array{Float64,1}
     omega          #::Array{Float64,1}
     Eqdist         #::Array{Float64,1}
     mass           #::Array{Float64,1}
     Z              #::Array{Float64,1}
     nocc           #::Array{Int64,1}
     function Atoms(Natoms, R, sigma, omega, Eqdist, mass, Z, nocc )
         return new(Natoms, R, sigma, omega, Eqdist, mass, Z, nocc )
     end
end
