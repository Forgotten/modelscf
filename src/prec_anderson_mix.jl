function prec_anderson_mix(vin, vout, beta, ymat, smat, iter::Int64, mixdim, prec, precargs)
    # function that computes the new potential usign the preconditioned anderson mix
    # TODO: This funcion should encapulate the normal anderson mix by providing
    #       a default preconditioner(such as the identity operator)
    # TODO: I think that the mixing can be efficiently be explained by a small
    # jupyter notebook

    n = length(vin);
    #
    # function evaluation overwrites vout
    #
    # residue
    res = vin - vout;
    #
    iterused = min(iter-1,mixdim);
    ipos = iter - 1 - round(Integer, floor((iter-2)/mixdim))*mixdim;
    #
    if (iter > 1)
       # compute the changes in function evaluations and the step (changes
       # in potentials)
       ymat[:,ipos] = res - ymat[:,ipos];
       smat[:,ipos] = vin - smat[:,ipos];
    end
    #
    vopt  = vin;
    ropt  = res;
    #
    if (iter > 1)
      # LLin: Solve the least square problem
       gammas = ymat[:,1:iterused] \ res;
       vopt   = vin - smat[:,1:iterused] * gammas;
       ropt   = res - ymat[:,1:iterused] * gammas;
    end

    inext = iter - round(Integer, floor((iter - 1) / mixdim)) * mixdim;
    ymat[:,inext] = res;
    smat[:,inext] = vin;

    resprec = prec(ropt, precargs);
    vnew = vopt - betamix*resprec;

    return (vnew, ymat, smat)
end
