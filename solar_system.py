#!/usr/bin/python3
# solar_system.py
"""Constraints on the stochastic GW background from the Solar System."""

__author__ = ("Alexander C. Jenkins",)
__contact__ = ("alex.jenkins@ucl.ac.uk",)
__version__ = "0.2"
__date__ = "2023/03"

import numpy as np

from scipy.optimize import root_scalar

from gwresonance import Binary, v_rms, NMAX, YR


#-----------------------------------------------------------------------------#


def dx_rms(sgwb,
           x_init,
           mp,
           msun=1.,
           lookback=4.5e+9 * YR,
           low_ecc=False,
           ):
    """RMS change in the orbital elements over a long timescale.

    This is given by summing the standard deviation and the shift in
    the mean in quadrature for each orbital element.

    Parameters
    ----------
    sgwb : float or function
        The (present-day) SGWB energy density spectrum, as a function
        of frequency (in Hertz). If ``sgwb'' is a float, the spectrum
        is assumed to be scale-invariant.
    x_init : float or array_like of shape (6,)
        Initial orbital elements of the binary. See Notes below for
        details.
    mp : float
        Mass of the planet, in solar units.
    msun : float, optional
        Mass of the star, in solar units.
    lookback : float, optional
        Cosmological lookback time to the moment when the binary is
        initialised, in seconds.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.

    Returns
    -------
    array
        Value of the RMS change for each orbital element.
    """

    b = Binary(sgwb,
               x_init,
               m1=msun,
               m2=mp,
               low_ecc=low_ecc,
               lookback=lookback,
               )
    b.evolve_fokker_planck(lookback,
                           t_eval=[0., lookback])

    return np.sqrt(b.dx[-1]**2. + b.cov[-1].diagonal())


def power_law_ul(alpha,
                 f_ref,
                 x_init,
                 mp,
                 msun=1.,
                 lookback=4.5e+9 * YR,
                 low_ecc=False,
                 bracket=(0., 11.),
                 ):
    r"""Calculate a Solar System upper limit on a power-law SGWB.

    The upper limit is set by requiring the RMS change in a planet's
    period, eccentricity and inclination over the lifetime of the Solar
    System to be less than their present-day values.

    Outputs an upper limit on :math:`\Omega_\mathrm{ref}`, where the
    SGWB spectrum is given by
    .. math::
        \Omega_\mathrm{gw}(f)
        = \Omega_\mathrm{ref} (f / f_\mathrm{ref})^\alpha.

    Parameters
    ----------
    alpha : float
        SGWB power-law index.
    f_ref : float
        SGWB reference frequency at which the upper limit is computed.
    x_init : array_like, shape (6,)
        Initial orbital elements of the planet.
    mp : float
        Mass of the planet in solar units.
    msun : float, optional
        Mass of the star in solar units.
    lookback : float, optional
        Age of the Solar System, in seconds.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    bracket : array_like, shape (2,), optional
        Bracket for the root-finding algorithm. The entries should be
        estimated maximum and minimum values of
        :math:`log_{10}\Omega_\mathrm{ref}`.

    Returns
    -------
    float
        Upper limit on :math:`\Omega_\mathrm{ref}`.
    """

    x_init = np.array(x_init)

    func = lambda logohm: np.max(
        dx_rms(lambda f: 10.**logohm * (f/f_ref)**alpha, x_init, mp,
               msun=msun, lookback=lookback, low_ecc=low_ecc)[:3] - x_init[:3])

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
             mp,
             msun=1.,
             lookback=4.5e+9 * YR,
             low_ecc=False,
             alphas=None,
             bracket=(0., 11.),
             verbose=False,
             ):
    """Calculate a planet's SGWB sensitivity curve.

    Returns the power-law integrated (PI) curve, as defined in [1]_.

    Parameters
    ----------
    freqs : array_like of floats
        Frequencies at which the sensitivity should be computed.
    x_init : array_like, shape (6,)
        Initial orbital elements of the planet.
    mp : float
        Mass of the planet in solar units.
    msun : float, optional
        Mass of the star in solar units.
    lookback : float, optional
        Age of the Solar System, in seconds.
    low_ecc : bool, optional
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    alphas : array_like of floats, optional
        Set of power-law indices which should be used to construct the
        sensitivity curve.
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
    f_ref = 1. / x_init[0]

    if verbose:
        print("Calculating PI curve...")

    for i, alpha in enumerate(alphas):
        if verbose:
            print("alpha = "+str(alpha))
        ul = power_law_ul(alpha, f_ref, x_init, mp, msun=msun,
                          lookback=lookback, low_ecc=low_ecc, bracket=bracket)
        curves[:, i] = ul * (freqs / f_ref) ** alpha

    return np.max(curves, -1)


#-----------------------------------------------------------------------------#
