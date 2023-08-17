#!/usr/bin/python3
# laser_ranging.py
"""Constraints on the stochastic GW background from laser ranging."""

__author__ = ("Alexander C. Jenkins",)
__contact__ = ("alex.jenkins@ucl.ac.uk",)
__version__ = "0.2"
__date__ = "2023/03"

import numpy as np

from scipy.optimize import root_scalar
from scipy.stats import chi2

from gwresonance import Binary, gamma, sma_from_per, v_rms, AU, NMAX


#-----------------------------------------------------------------------------#


def fisher(x,
           t_obs,
           n_obs,
           sigma=3.e-3,
           m1=3.00345e-06,
           m2=3.69223e-08,
           ):
    """Laser-ranging Fisher matrix for measuring orbital elements.

    Parameters
    ----------
    x : array_like, shape (6,)
        Orbital elements of the binary.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of normal point observations recorded per
        observation period.
    sigma : float, optional
        RMS distance noise for each normal point, in metres. Default is
        3mm.
    m1 : float, optional
        Mass of the primary body in solar units. Default is the Earth's
        mass.
    m2 : float, optional
        Mass of the secondary body in solar units. Default is the
        Moon's mass.

    Returns
    -------
    array, shape (6,6)
        Fisher matrix for the orbital elements.
    """

    out = np.zeros((6, 6))
    (per, ecc, inc, asc, arg, eps) = x

    if ecc == None:
        out[0, 0] = 4. / 9. / per**2.

    else:
        g = gamma(ecc)

        out[0, 0] = (4. / 9. / per**2. * (1. + 3.*ecc + 27./16.*ecc**2.
                                          + 4.*ecc**3. + 315./256.*ecc**4.)
                     + 2. / 3. * (np.pi * ecc * t_obs)**2. / per**4.
                     * (1. + 0.25*ecc**2.))
        out[0, 1] = 0.75 / per * ecc * (1. + 8./9.*ecc + 5./8.*ecc**2.
                                        + 8./45.*ecc**3.)
        out[1, 0] = out[0, 1]
        out[1, 1] = 2. - g - (1. - g) / ecc**2.
        out[5, 5] = 1. - g

    return out * n_obs * (sma_from_per(per, m1+m2) * AU / sigma) ** 2.


def likelihood_ratio(sgwb,
                     x_init,
                     times,
                     t_obs,
                     n_obs,
                     sigma=3.e-3,
                     m1=3.00345e-06,
                     m2=3.69223e-08,
                     exclude_elements=None,
                     ):
    """Calculate the likelihood ratio statistic for a laser-ranged
    binary.

    Parameters
    ----------
    sgwb : float or function
        SGWB energy density spectrum. If ``sgwb'' is a float, the
        spectrum is assumed to be scale-invariant.
    x_init : array_like, shape (6,)
        Initial orbital elements of the binary.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of normal point observations recorded per
        observation period.
    sigma : float, optional
        RMS distance noise for each normal point, in metres. Default is
        3mm.
    m1 : float, optional
        Mass of the primary body in solar units. Default is the Earth's
        mass.
    m2 : float, optional
        Mass of the secondary body in solar units. Default is the
        Moon's mass.
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
               m1=m1,
               m2=m2,
               )
    b.evolve_fokker_planck(times[-1],
                           t_eval=times)

    out = 0.

    if exclude_elements is None:
        exclude_elements = np.array([True]*2 +[False]*2 +[True] +[False])

    ex = exclude_elements

    for t in times:
        x0 = b.x0[b.t == t][0]
        dx = b.dx[b.t == t][0][ex]
        cov = b.cov[b.t == t][0][:,ex][ex,:]
        fish = fisher(x0, t_obs, n_obs, sigma=sigma, m1=m1, m2=m2)[:,ex][ex,:]
        out += np.linalg.multi_dot([dx, fish, dx])
        out += np.dot(fish, cov).trace()
        (s, logdet) = np.linalg.slogdet(np.eye(len(dx)) + np.dot(fish, cov))
        out -= s * logdet

    return out


def power_law_ul(alpha,
                 f_ref,
                 x_init,
                 times,
                 t_obs,
                 n_obs,
                 sigma=3.e-3,
                 m1=3.00345e-06,
                 m2=3.69223e-08,
                 confidence=0.95,
                 exclude_elements=None,
                 bracket=(-20., +20.),
                 ):
    r"""Calculate a laser-ranging upper limit on a power-law SGWB.

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
        Initial orbital elements of the binary.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of normal point observations recorded per
        observation period.
    sigma : float, optional
        RMS distance noise for each normal point, in metres. Default is
        3mm.
    m1 : float, optional
        Mass of the primary body in solar units. Default is the Earth's
        mass.
    m2 : float, optional
        Mass of the secondary body in solar units. Default is the
        Moon's mass.
    confidence : float, optional
        Set the p-value of the upper limit.
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

    func = (lambda logohm:
            likelihood_ratio(lambda f: 10.**logohm * (f/f_ref)**alpha,
                             x_init, times, t_obs, n_obs, sigma=sigma,
                             m1=m1, m2=m2, exclude_elements=exclude_elements)
            - chi2_crit)

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
             times,
             t_obs,
             n_obs,
             sigma=3.e-3,
             m1=3.00345e-06,
             m2=3.69223e-08,
             confidence=0.95,
             alphas=None,
             exclude_elements=None,
             bracket=(-20., +20.),
             verbose=False,
             ):
    """Calculate a laser-ranged binary's SGWB sensitivity curve.

    Returns the power-law integrated (PI) curve, as defined in [1]_.

    Parameters
    ----------
    freqs : array_like of floats
        Frequencies at which the sensitivity should be computed.
    x_init : array_like, shape (6,) or (N, 6)
        Initial orbital elements of the binary.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of normal point observations recorded per
        observation period.
    sigma : float, optional
        RMS distance noise for each normal point, in metres. Default is
        3mm.
    m1 : float, optional
        Mass of the primary body in solar units. Default is the Earth's
        mass.
    m2 : float, optional
        Mass of the secondary body in solar units. Default is the
        Moon's mass.
    confidence : float, optional
        Set the p-value of the upper limit.
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
    f_ref = 1. / x_init[0]

    if verbose:
        print("Calculating PI curve...")

    for i, alpha in enumerate(alphas):
        if verbose:
            print("alpha = "+str(alpha))
        ul = power_law_ul(alpha, f_ref, x_init, times, t_obs, n_obs,
                          sigma=sigma, m1=m1, m2=m2, confidence=confidence,
                          exclude_elements=exclude_elements, bracket=bracket)
        curves[:, i] = ul * (freqs / f_ref) ** alpha

    return np.max(curves, -1)


def comb_ul(x_init,
            times,
            t_obs,
            n_obs,
            sigma=3.e-3,
            m1=3.00345e-06,
            m2=3.69223e-08,
            confidence=0.95,
            alphas=None,
            exclude_elements=None,
            bracket=(-20., +20.),
            verbose=False,
            ):
    """Calculate a laser-ranged binary's SGWB sensitivity at each
    harmonic.

    Returns the SGWB upper limit corresponding to the discrete set of
    frequencies ``[f0, 2*f0, 3*f0, ..., nmax*f0]'', where ``f0'' is the
    base frequency (inverse of the binary period), and ``nmax'' is the
    maximum harmonic (usually 20, but may be less for short periods).

    Parameters
    ----------
    x_init : array_like, shape (6,)
        Initial orbital elements of the binary.
    times : array_like
        Array of times at which the binary orbital elements are
        measured, in seconds.
    t_obs : float
        Observation period over which each measurement of the binary
        orbital elements is performed, in seconds.
    n_obs : float
        Mean number of normal point observations recorded per
        observation period.
    sigma : float, optional
        RMS distance noise for each normal point, in metres. Default is
        3mm.
    m1 : float, optional
        Mass of the primary body in solar units. Default is the Earth's
        mass.
    m2 : float, optional
        Mass of the secondary body in solar units. Default is the
        Moon's mass.
    confidence : float, optional
        Set the p-value of the upper limit.
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
        The SGWB upper limit at the first few of the binary's harmonic
        frequencies.
    """

    per = x_init[0]
    f0 = 1. / per
    nmax = min(int(np.floor(v_rms(per, m1+m2) ** -1.)), NMAX)
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
                             x_init, times, t_obs, n_obs, sigma=sigma,
                             m1=m1, m2=m2, exclude_elements=exclude_elements)
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
