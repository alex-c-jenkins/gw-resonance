#!/usr/bin/python3
# binary_pulsar.py
"""Constraints on the stochastic GW background from binary pulsars."""

__author__ = ("Alexander C. Jenkins",)
__contact__ = ("alex.jenkins@ucl.ac.uk",)
__version__ = "0.2"
__date__ = "2023/03"

import numpy as np

from scipy.optimize import root_scalar
from scipy.stats import chi2

from gwresonance import Binary, gamma, v_rms, NMAX


#-----------------------------------------------------------------------------#


def fisher(x,
           mc,
           t_obs,
           n_obs,
           mp=1.35,
           sigma=80.e-9,
           low_ecc=False,
           ):
    """Pulsar-timing Fisher matrix for measuring orbital elements.

    Based on the Blandford-Teukolsky timing formula [1]_.

    Parameters
    ----------
    x : array_like, shape (6,)
        Orbital elements of the binary pulsar.
    mc : float
        Mass of the companion in solar units.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of pulse time-of-arrival integrations made per
        observation period ``t_obs''.
    mp : float, optional
        Mass of the pulsar in solar units.
    sigma : float, optional
        RMS timing noise for each of the integrations, in seconds.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.

    Returns
    -------
    array, shape (6,6)
        Fisher matrix for the orbital elements.

    References
    ----------
    .. [1] Roger Blandford and Saul A. Teukolsky, “Arrival-time
        analysis for a pulsar in a binary system,” Astrophys. J. 205,
        580–591 (1976).
    """

    out = np.zeros((6, 6))

    if not low_ecc:
        (per, ecc, inc, asc, arg, eps) = x

        if ecc == None:
            out[0, 0] = (np.cos(2.*arg) / per**2.
                         + 2. * np.pi * t_obs * np.sin(2.*arg) / per**3.
                         + 8. / 3. * (np.pi*t_obs)**2. / per**4.)
            out[0, 2] = np.cos(2.*arg) / np.tan(inc) / per
            out[2, 0] = out[0, 2]
            out[0, 4] = -np.sin(2.*arg) / per - 2. * np.pi * t_obs / per**2.
            out[4, 0] = out[0, 4]
            out[0, 5] = -np.sin(2.*arg) / per - 2. * np.pi * t_obs / per**2.
            out[5, 0] = out[0, 5]
            out[2, 2] = 2 / np.tan(inc)**2.
            out[4, 4] = 2.
            out[4, 5] = 2.
            out[5, 5] = 2.

        else:
            gg = gamma(ecc)
            f1 = (1.
                  + 8. / 9. * ecc
                  - 3. / 16. * ecc**2.
                  - 448. / 225. * ecc**3.
                  - 175. / 288. * ecc**4.
                  - 11584. / 11025. * ecc**5.
                  - 2975. / 9216. * ecc**6.
                  - 67264. / 99225. * ecc**7.
                  - 96733. / 460800. * ecc**8.
                  - 5818432. / 12006225. * ecc**9.
                  - 278579. / 1843200. * ecc**10.
                  - 149726912. / 405810405. * ecc**11.
                  - 20910823. / 180633600. * ecc**12.)
            f2 = (1.
                  + 4. / 3. * ecc
                  + 5. / 8. * ecc**2.
                  + 2. / 15. * ecc**3.
                  + 1. / 48. * ecc**4.
                  + 1. / 210. * ecc**5.
                  - 29. / 768. * ecc**6.
                  - 31. / 1260. * ecc**7.
                  - 359. / 7680. * ecc**8.
                  - 3559. / 110880. * ecc**9.
                  - 469. / 10240. * ecc**10.
                  - 19087. / 576576. * ecc**11.
                  - 6099. / 143360. * ecc**12.)
            f3 = (ecc**2.
                  + 1. / 2. * ecc**4.
                  + 5. / 16. * ecc**6.
                  + 7. / 32. * ecc**8.
                  + 21. / 128. * ecc**10.
                  + 33. / 256. * ecc**12.)

            out[0, 0] = (8. * ecc * (1. + 0.25*ecc**2.) / per**2.
                         + np.cos(2.*arg) * f1 / per**2.
                         + 2. * np.pi * t_obs * np.sin(2.*arg) * f2 / per**3.
                         + 8. / 3. * (np.pi*t_obs)**2.
                         * (1. - 0.25*np.cos(2.*arg)*f3) / per**4.)
            out[0, 1] = (4. * (1. + 0.375*ecc) * (1. - 1./6.*ecc**2.) / per
                         + np.pi * t_obs * ecc * np.sin(2.*arg) / gg / per**2.
                         - 8. / 3. * np.cos(2.*arg) / per
                         * (1. + 0.75*ecc + 1./6.*ecc**2. + 1./16.*ecc**3.))
            out[1, 0] = out[0, 1]
            out[0, 2] = (2. * ecc * (1. + 0.25*ecc)
                         + np.cos(2.*arg)
                         * (1. - 2.*ecc - 1.5*ecc**2.)) / np.tan(inc) / per
            out[2, 0] = out[0, 2]
            out[0, 4] = -(np.sin(2.*arg) * (1. - 2.*ecc - 1.5*ecc**2.) / per
                          + 2. * np.pi * t_obs * gg / per**2.)
            out[4, 0] = out[0, 4]
            out[0, 5] = -(np.sin(2.*arg) * f2 / per
                          + 2. * np.pi * t_obs
                          * (1. - 0.25*np.cos(2.*arg)*f3) / per**2.)
            out[5, 0] = out[0, 5]
            out[1, 1] = (5. / gg**2.
                         * (1. - 0.75*ecc**2. - 0.05*ecc**4.
                            - 0.9*np.cos(2.*arg)
                            * (1. - 23./18.*ecc**2. + 1./18.*ecc**4.)))
            out[1, 2] = 3. * ecc / np.tan(inc) * (1. + 1./12.*ecc**2.
                                                  - 11./6.*np.cos(2.*arg)
                                                  * (1. - 1./22.*ecc**2.))
            out[2, 1] = out[1, 2]
            out[1, 4] = 5.5 * ecc * np.sin(2.*arg) * (1. - 1./22.*ecc**2.)
            out[4, 1] = out[1, 4]
            out[1, 5] = ecc / gg * np.sin(2.*arg)
            out[5, 1] = out[1, 5]
            out[2, 2] = 2. / np.tan(inc)**2. * (1. + 1.5*ecc**2.
                                                - 2.5*ecc**2.*np.cos(2.*arg))
            out[2, 4] = 5. * ecc**2. / np.tan(inc) * np.sin(2.*arg)
            out[4, 2] = out[2, 4]
            out[4, 4] = 2. + 3.*ecc**2. + 5.*ecc**2.*np.cos(2.*arg)
            out[4, 5] = 2. * gg
            out[5, 4] = out[4, 5]
            out[5, 5] = 4. * (np.sin(arg)**2. + gg*np.cos(arg)**2.) / (1.+gg)

    if low_ecc:
        (per, zet, kap, inc, asc, xxi) = x
        ecc = (zet**2. + kap**2.) ** 0.5

        out[0, 0] = (((kap**2. - zet**2.) / ecc**2.
                      + 4. * np.pi * t_obs / per * zet * kap / ecc**2.
                      + 8. / 3. * (np.pi*t_obs/per)**2.) / per**2.)
        out[0, 1] = 4. / 3. * zet / per * (5.*zet**2. + 6.*kap**2.) / ecc**3.
        out[1, 0] = out[0, 1]
        out[0, 2] = 4. / 3. * kap / per * kap**2. / ecc**3.
        out[2, 0] = out[0, 2]
        out[0, 3] = (kap**2. - zet**2.) / per / ecc**2. / np.tan(inc)
        out[3, 0] = out[0, 3]
        out[0, 5] = -2. / per * (zet * kap / ecc**2. + np.pi * t_obs / per)
        out[5, 0] = out[0, 5]
        out[1, 1] = 9.5
        out[1, 2] = 0.75 * zet * kap
        out[2, 1] = out[1, 2]
        out[1, 3] = zet * (7.5 + (zet/ecc)**2.) / np.tan(inc)
        out[3, 1] = out[1, 3]
        out[1, 5] = 2. * kap * (zet**2. - kap**2.) / ecc**2.
        out[5, 1] = out[1, 5]
        out[2, 2] = 0.5
        out[2, 3] = kap * (-2.5 + (zet/ecc)**2.) / np.tan(inc)
        out[3, 2] = out[2, 3]
        out[2, 5] = zet * (2.5 - (zet/ecc)**2.)
        out[5, 2] = out[2, 5]
        out[3, 3] = 2. / np.tan(inc)**2.
        out[5, 5] = 2.

    return out * n_obs * (per * v_rms(per, mp+mc) * np.sin(inc)
                          / 4. / np.pi / sigma / (1. + mp/mc)) ** 2.


def likelihood_ratio(sgwb,
                     x_init,
                     mc,
                     times,
                     t_obs,
                     n_obs,
                     mp=1.35,
                     sigma=80.e-9,
                     low_ecc=False,
                     exclude_elements=None,
                     ):
    """Calculate the likelihood ratio statistic for a binary pulsar.

    Parameters
    ----------
    sgwb : float or function
        SGWB energy density spectrum. If ``sgwb'' is a float, the
        spectrum is assumed to be scale-invariant.
    x_init : array_like, shape (6,)
        Initial orbital elements of the binary pulsar.
    mc : float
        Mass of the companion in solar units.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of pulse time-of-arrival integrations made per
        observation period t_obs.
    mp : float, optional
        Mass of the pulsar in solar units.
    sigma : float, optional
        RMS timing noise for each of the integrations, in seconds.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    exclude_elements : array_like, shape (6,), optional
        Exclude orbital elements from the calculation by setting the
        corresponding elements of this array to ``False'', and the
        rest to ``True''. Default depends on the value of ``low_ecc''.

    Returns
    -------
    float
        Value of the maximum likelihood ratio statistic.
    """

    b = Binary(sgwb,
               x_init,
               m1=mp,
               m2=mc,
               low_ecc=low_ecc,
               )
    b.evolve_fokker_planck(times[-1],
                           t_eval=times)

    out = 0.

    if exclude_elements is None:
        if not low_ecc:
            exclude_elements = np.array([True]*3 + [False] + [True]*2)

        if low_ecc:
            exclude_elements = np.array([True]*4 + [False] + [True])

    ex = exclude_elements

    for t in times:
        x0 = b.x0[b.t == t][0]
        dx = b.dx[b.t == t][0][ex]
        cov = b.cov[b.t == t][0][:, ex][ex, :]
        fish = fisher(x0, mc, t_obs, n_obs, mp=mp, sigma=sigma,
                      low_ecc=low_ecc)[:, ex][ex, :]
        out += np.linalg.multi_dot([dx, fish, dx])
        out += np.dot(fish, cov).trace()
        (s, logdet) = np.linalg.slogdet(np.eye(len(dx)) + np.dot(fish, cov))
        out -= s * logdet

    return out


def power_law_ul(alpha,
                 f_ref,
                 x_init,
                 mc,
                 times,
                 t_obs,
                 n_obs,
                 mp=1.35,
                 sigma=80.e-9,
                 confidence=0.95,
                 low_ecc=False,
                 exclude_elements=None,
                 bracket=(-20., +20.),
                 ):
    r"""Calculate a pulsar-timing upper limit on a power-law SGWB.

    Outputs a forecast upper limit on :math:`\Omega_\mathrm{ref}`,
    where the SGWB spectrum is given by
    .. math::
        \Omega_\mathrm{gw}(f)
        = \Omega_\mathrm{ref} (f / f_\mathrm{ref})^\alpha.

    Parameters
    ----------
    alpha : float
        SGWB power-law index.
    f_ref : float
        SGWB reference frequency at which the upper limit is computed.
    x_init : array_like, shape (6,) or (N, 6)
        Initial orbital elements of the binary pulsar(s).
    mc : float or array_like, shape (N,)
        Mass(es) of the companion(s) in solar units.
    times : array_like, shape (_,) or (N, _)
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float or array_like, shape (N,)
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float or array_like, shape (N,)
        Mean number of pulse time-of-arrival integrations made per
        observation period t_obs.
    mp : float or array_like, shape (N,), optional
        Mass(es) of the pulsar(s) in solar units.
    sigma : float or array_like, shape (N,), optional
        RMS timing noise for each of the integrations, in seconds.
    confidence : float, optional
        Set the p-value of the upper limit.
    low_ecc : bool or array_like, shape (N,), optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    exclude_elements : array_like, shape (6,), optional
        Exclude orbital elements from the calculation by setting the
        corresponding elements of this array to ``False'', and the
        rest to ``True''.
    bracket : array_like, shape (2,), optional
        Bracket for the root-finding algorithm. The entries should be
        estimated maximum and minimum values of
        :math:`log_{10}\Omega_\mathrm{ref}`.

    Returns
    -------
    float
        Upper limit on :math:`\Omega_\mathrm{ref}`.
    """

    chi2_crit = chi2.ppf(confidence, 1)
    x_init = np.array(x_init)
    mc = np.array(mc)
    times = np.array(times)
    t_obs = np.array(t_obs)
    n_obs = np.array(n_obs)
    mp = np.array(mp)
    sigma = np.array(sigma)
    low_ecc = np.array(low_ecc)

    if x_init.shape != (6,):
        N = x_init.shape[0]

        if mc.ndim == 0:
            mc = mc * np.ones(N)

        if times[0].ndim == 0:
            times = np.outer(np.ones(N), times)

        if t_obs.ndim == 0:
            t_obs = t_obs * np.ones(N)

        if n_obs.ndim == 0:
            n_obs = n_obs * np.ones(N)

        if mp.ndim == 0:
            mp = mp * np.ones(N)

        if sigma.ndim == 0:
            sigma = sigma * np.ones(N)

        if low_ecc.ndim == 0:
            low_ecc = low_ecc * np.ones(N)

        func = (lambda logohm: np.sum(
            [likelihood_ratio(lambda f: 10.**logohm * (f/f_ref)**alpha,
                              x_init[i], mc[i], times[i], t_obs[i], n_obs[i],
                              mp=mp[i], sigma=sigma[i], low_ecc=low_ecc[i])
             for i in range(N)]) - chi2_crit)

    else:
        func = (lambda logohm:
            likelihood_ratio(lambda f: 10.**logohm * (f/f_ref)**alpha,
                             x_init, mc, times, t_obs, n_obs, mp=mp,
                             sigma=sigma, low_ecc=low_ecc) - chi2_crit)

    if func(bracket[1]) < 0.:
        print("Warning: upper limit lies above bracket.")
        return 10. ** bracket[1]

    elif func(bracket[0]) > 0.:
        print("Warning: upper limit lies below bracket.")
        return 10. ** bracket[0]

    else:
        sol = root_scalar(func,
                          method="brentq",
                          bracket=bracket,
                          )
        return 10. ** sol.root


def pi_curve(freqs,
             x_init,
             mc,
             times,
             t_obs,
             n_obs,
             mp=1.35,
             sigma=80.e-9,
             confidence=0.95,
             low_ecc=False,
             alphas=None,
             exclude_elements=None,
             bracket=(-20., +20.),
             verbose=False,
             ):
    """Calculate a binary pulsar's SGWB sensitivity curve.

    Returns the power-law integrated (PI) curve, as defined in [1]_.

    Parameters
    ----------
    freqs : array_like of floats
        Frequencies at which the sensitivity should be computed.
    x_init : array_like, shape (6,) or (N, 6)
        Initial orbital elements of the binary pulsar(s).
    mc : float or array_like, shape (N,)
        Mass(es) of the companion(s) in solar units.
    times : array_like, shape (_,) or (N, _)
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float or array_like, shape (N,)
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float or array_like, shape (N,)
        Mean number of pulse time-of-arrival integrations made per
        observation period t_obs.
    mp : float or array_like, shape (N,), optional
        Mass(es) of the pulsar(s) in solar units.
    sigma : float or array_like, shape (N,), optional
        RMS timing noise for each of the integrations, in seconds.
    confidence : float, optional
        Set the p-value of the upper limit.
    low_ecc : bool or array_like, shape (N,), optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    alphas : array_like of floats, optional
        Set of power-law indices which should be used to construct the
        sensitivity curve.
    exclude_elements : array_like, shape (6,), optional
        Exclude orbital elements from the calculation by setting the
        corresponding elements of this array to ``False'', and the
        rest to ``True''.
    bracket : array_like, shape (2,), optional
        Bracket for the root-finding algorithm. The entries should be
        estimated maximum and minimum values of
        :math:`log_{10}\Omega_\mathrm{ref}`.
    verbose : bool, optional
        Control the level of output.

    Returns
    -------
    array of floats
        The value of the sensitivity curve at each of the frequencies
        in ``freqs''.

    References
    ----------
    .. [1] Eric Thrane and Joseph D. Romano, "Sensitivity curves for
        searches for gravitational-wave backgrounds," Phys. Rev. D 88,
        124032 (2013).
    """

    if alphas is None:
        alphas = np.linspace(-10., 10., 81)

    nf = len(freqs)
    na = len(alphas)
    curves = np.zeros((nf, na))
    x_init = np.array(x_init)

    if x_init.shape == (6,):
        f_ref = 1. / x_init[0]

    else:
        f_ref = np.min(1. / x_init[:, 0])

    if verbose:
        print("Calculating PI curve...")

    for i, alpha in enumerate(alphas):
        if verbose:
            print("alpha = "+str(alpha))
        ul = power_law_ul(alpha, f_ref, x_init, mc, times, t_obs, n_obs, mp=mp,
                          sigma=sigma, confidence=confidence, low_ecc=low_ecc,
                          exclude_elements=exclude_elements, bracket=bracket)
        curves[:, i] = ul * (freqs / f_ref) ** alpha

    return np.max(curves, -1)


def comb_ul(x_init,
            mc,
            times,
            t_obs,
            n_obs,
            mp=1.35,
            sigma=80.e-9,
            confidence=0.95,
            low_ecc=False,
            exclude_elements=None,
            bracket=(-20., +20.),
            verbose=False,
            ):
    """Calculate a binary pulsar's SGWB sensitivity at each harmonic.

    Returns the SGWB upper limit corresponding to the discrete set of
    frequencies ``[f0, 2*f0, 3*f0, ..., nmax*f0]'', where ``f0'' is the
    base frequency (inverse of the binary period), and ``nmax'' is the
    maximum harmonic (usually 20, but may be less for short periods).

    Parameters
    ----------
    x_init : array_like, shape (6,)
        Initial orbital elements of the binary pulsar.
    mc : float
        Mass of the companion in solar units.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of pulse time-of-arrival integrations made per
        observation period t_obs.
    mp : float, optional
        Mass of the pulsar in solar units.
    sigma : float, optional
        RMS timing noise for each of the integrations, in seconds.
    confidence : float, optional
        Set the p-value of the upper limit.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    exclude_elements : array_like, shape (6,), optional
        Exclude orbital elements from the calculation by setting the
        corresponding elements of this array to ``False'', and the
        rest to ``True''.
    bracket : array_like, shape (2,), optional
        Bracket for the root-finding algorithm. The entries should be
        estimated maximum and minimum values of
        :math:`log_{10}\Omega_\mathrm{ref}`.
    verbose : bool, optional
        Control the level of output.

    Returns
    -------
    array of floats
        The SGWB upper limit at the first few of the binary's harmonic
        frequencies.
    """

    per = x_init[0]
    f0 = 1. / per
    nmax = min(int(np.floor(v_rms(per, mp+mc) ** -1.)), NMAX)
    chi2_crit = chi2.ppf(confidence, 1)
    comb = []

    if verbose:
        print("Calculating comb...")

    for n in range(1, nmax+1):
        if verbose:
            print("Harmonic {} of {}".format(n, nmax))
        func = (lambda logohm:
            likelihood_ratio(lambda f: 10.**logohm
                             * np.heaviside(0.1*f0 - abs(f-n*f0), 0.),
                             x_init, mc, times, t_obs, n_obs, mp=mp,
                             sigma=sigma, low_ecc=low_ecc,
                             exclude_elements=exclude_elements)
            - chi2_crit)

        if func(bracket[1]) < 0.:
            print("Warning: upper limit lies above bracket.")
            comb.append(10. ** bracket[1])

        elif func(bracket[0]) > 0.:
            print("Warning: upper limit lies below bracket.")
            comb.append(10. ** bracket[0])

        else:
            sol = root_scalar(func,
                              method="brentq",
                              bracket=bracket,
                              )
            comb.append(10. ** sol.root)

    return np.array(comb)


#-----------------------------------------------------------------------------#
