function anderson_mix(vin, vout, beta, ymat, smat, iter, mixdim)
    # function that computes the new potential usign the anerson mix

    n=length(vin);
    #
    # function evaluation overwrites vout
    #
    res = vin - vout;
    #
    iterused = min(iter-1,mixdim);
    ipos = iter - 1 - floor((iter-2)/mixdim)*mixdim;
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

    inext = iter - floor((iter - 1) / mixdim) * mixdim;
    ymat[:,inext] = res;
    smat[:,inext] = vin;

    vnew = vopt - beta*ropt;

    return (vnew, ymat, smat)
end
