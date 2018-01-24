struct scfOptions
    # structure to encapsulate all the different options for the scf iterations
    # the main idea is to define a Hamiltonian, and then run scf on it; however,
    # the Hamiltoinian shouldn't have the scf options embeded
    mixtime::String
    betamix::Float64
    mixdim::Int64
    SCFtol::Float64
    scfiter::Int64
    eigstol::Float64
    eigsiter::Int64
    eigmethod::AbstractString

    function scfOptions()
        new("anderson",0.5, 10, 1e-7,100,1e-8, 100, "eigs")
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
        new(1e-8, 100, "eigs")
    end

    function eigOptions(opts::scfOptions)
        new(opts.eigstol, opts.eigsiter, opts.eigmethod)
    end
end

#abstract struct mixingOptions end

mutable struct andersonMixOptions # <: mixingOptions
    ymat::Array{Float64,2}
    smat::Array{Float64,2}
    betamix::Float64
    mixdim::Int64
    iter::Int64
    function andersonMixOptions(Ns, scfOpts::scfOptions)
        ymat = zeros(Ns, scfOpts.mixdim);
        smat = zeros(Ns, scfOpts.mixdim);
        new(ymat,smat,scfOpts.betamix[1],scfOpts.mixdim,1)
    end
end

function updateMix!( mixOpts::andersonMixOptions, ii )
    mixOpts.iter = ii;
end
