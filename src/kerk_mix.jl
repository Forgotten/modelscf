function  = kerk_mix(res, kerkMixOptions)
#The Kerker mixing, following the parameter setup of elliptic
#preconditioner.
#
#Lin Lin
#Last revision: 5/6/2012

KerkerB    = kerkMixOptions.KerkerB;
kx         = kerkMixOptions.kx;
YukawaK    = kerkMixOptions.YukawaK;
epsil0     = kerkMixOptions.epsil0;

rfft = fft(res);
gkk2 = kx.^2 ;

rfft = ( epsil0 / (4*pi) * (gkk2+YukawaK^2) ) ./ ...
  ( KerkerB + epsil0 / (4*pi) * (gkk2+YukawaK^2) ) .* rfft;

resprec = ifft(rfft);
resprec = resprec - mean(resprec);

return resprec
