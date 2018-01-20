struct scfOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    mixtime::String
    betamix::Float64
    SCFtol::Float64
    scfiter::Int64

    funtion scfOptions()
    new scfOptions('anderson',0.5,1e-7,100)
end
