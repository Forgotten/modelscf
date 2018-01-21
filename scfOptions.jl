struct scfOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    mixtime::String
    betamix::Float64
    SCFtol::Float64
    scfiter::Int64
    eigstol::Float64
    eigsiter::Int64
    eigmethod::AbstractString

    function scfOptions()
        new scfOptions('anderson',0.5,1e-7,100,1e-9, 50, "eigs")
    end
end

struct eigOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    eigstol::Float64
    eigsiter::Int64
    eigmethod::AbstractString

    function eigOptions()
        new scfOptions(1e-9, 50, "eigs")
    end

    function eigOptions(opts::scfOptions)
        new scfOptions(opts.eigstol, opts.eigsiter, opt.eigmethod)
    end
end
