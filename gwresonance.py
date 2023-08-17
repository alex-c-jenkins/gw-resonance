#!/usr/bin/python3
# gwresonance.py
"""Binary orbit evolution due to the stochastic GW background."""

__author__ = ("Alexander C. Jenkins",)
__contact__ = ("alex.jenkins@ucl.ac.uk",)
__version__ = "0.2"
__date__ = "2023/03"

import matplotlib.pyplot as plt
import numpy as np

from inspect import isfunction
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root_scalar
from scipy.special import factorial, hyp2f1, jn, poch
from tqdm import tqdm

from interpolated_function import InterpolatedFunction


#-----------------------------------------------------------------------------#
# Physical constants.


YR = 3.15576e+07             # Year in seconds
DAY = 8.64000e+04            # Day in seconds
AU = 149597870700.           # Astronomical unit in metres
H100 = 0.6766                # Hubble constant in units of 100 km/s/Mpc,
                             #     Planck 2018 value
HUB0 = H100 * 3.240779e-18   # Hubble constant, H_0, in Hertz
MAT0 = 0.3111                # Present-day matter density fraction,
                             #     Planck 2018 value
NEWT = 4.92549094830932e-06  # Newton's constant over speed of light
                             #     cubed in seconds per solar mass
C_AU = 2.0039888041e-03      # Speed of light in AU per second


#-----------------------------------------------------------------------------#
# Internal constants.


NMAX = 20        # Maximum harmonic implemented in ``km_funcs''
ECC_MIN = 0.001  # Minimum eccentricity implemented in ``km_funcs''
ECC_MAX = 0.999  # Maximum eccentricity implemented in ``km_funcs''


#-----------------------------------------------------------------------------#
# SGWB utility functions.


def ohm_gw_from_h_c(h_c, freq):
    """Convert SGWB characteristic strain to density parameter.

    Parameters
    ----------
    h_c : float or array_like of floats
        The characteristic strain amplitude.
    freq : float or array_like of floats
        The GW frequency in Hz.

    Returns
    -------
    float or array
        Value of the SGWB density parameter.
    """

    return 2. / 3. * (np.pi / HUB0 * freq * h_c) ** 2.


def h_c_from_ohm_gw(ohm_gw, freq):
    """Convert SGWB density parameter to characteristic strain.

    Parameters
    ----------
    ohm_gw : float or array_like of floats
        The SGWB energy density parameter.
    freq : float or array_like of floats
        The GW frequency in Hz.

    Returns
    -------
    float or array
        Value of the characteristic strain amplitude.
    """

    return (1.5*ohm_gw) ** 0.5 * HUB0 / np.pi / freq


#-----------------------------------------------------------------------------#
# Binary utility functions


def sma_from_per(per, mass):
    """Convert binary period to semi-major axis.

    Parameters
    ----------
    per : float or array_like of floats
        Period of the binary in seconds.
    mass : float or array_like of floats
        Total mass of the binary in solar units.

    Returns
    -------
    float or array
        Semi-major axis of the binary in AU.
    """

    return C_AU * (NEWT * mass) ** (1./3.) * (0.5 * per / np.pi) ** (2./3.)


def per_from_sma(sma, mass):
    """Convert binary semi-major axis to period.

    Parameters
    ----------
    sma : float or array_like of floats
        Semi-major axis of the binary in astronomical units.
    mass : float or array_like of floats
        Total mass of the binary in solar units.

    Returns
    -------
    float or array
        Period of the binary in seconds.
    """

    return 2.0 * np.pi * (sma/C_AU) ** 1.5 * (NEWT*mass) ** -0.5


def v_rms(per, mass):
    """Calculate the root-mean-square orbital velocity of a binary.

    Parameters
    ----------
    per : float or array_like of floats
        Period of the binary in seconds.
    mass : float or array_like of floats
        Total mass of the binary in solar units.

    Returns
    -------
    float or array
        RMS orbital velocity of the binary, as a fraction of the speed
        of light.
    """

    return (2. * np.pi * NEWT * mass / per) ** (1./3.)


#-----------------------------------------------------------------------------#
# Cosmological functions


def lookback(z):
    """Cosmological lookback time as a function of redshift.

    Parameters
    ----------
    z : float
        Cosmological redshift.

    Returns
    -------
    float
        Value of the lookback time in seconds.
    """

    return quad(lambda x: (1.+x) ** -1. * (1.-MAT0 + MAT0*(1.+x)**3.) ** -0.5,
                0., z)[0] / HUB0


def redshift(t):
    """Cosmological redshift as a function of lookback time.

    Parameters
    ----------
    t : float
        Lookback time in seconds.

    Returns
    -------
    float
        Value of the cosmological redshift.
    """

    func = lambda z: lookback(z) - t
    sol = root_scalar(func,
                      method="bisect",
                      bracket=(0., 10.),
                      xtol=0.001)

    return sol.root


redshift = InterpolatedFunction(redshift,
                                np.linspace(0., 0.9/HUB0, 100000))
redshift.interpolate()


#-----------------------------------------------------------------------------#
# Eccentricity functions


def gamma(ecc):
    """Calculate the dimensionless angular momentum of a binary.

    Parameters
    ----------
    ecc : float or array_like of floats
        Binary eccentricity.

    Returns
    -------
    float or array
        Dimensionless angular momentum of the binary.
    """

    return (1. - ecc**2.) ** 0.5


def beta(ecc):
    """Calculate the Hansen-coefficient expansion parameter.

    Parameters
    ----------
    ecc : float or array_like of floats
        Binary eccentricity.

    Returns
    -------
    float or array
        Value of the eccentric parameter.
    """

    return ecc / (1.+gamma(ecc))


def gammap(ecc):
    r"""Calculate the eccentricity derivative of :math:`\gamma`.

    Parameters
    ----------
    ecc : float or array_like of floats
        Binary eccentricity.

    Returns
    -------
    float or array
        Value of the derivative.
    """

    return -ecc / gamma(ecc)


def betap(ecc):
    r"""Calculate the eccentricity derivative of :math:`\beta`.

    Parameters
    ----------
    ecc : float or array_like of floats
        Binary eccentricity.

    Returns
    -------
    float or array
        Value of the derivative.
    """

    return (gamma(ecc) * (1.+gamma(ecc))) ** -1.


#-----------------------------------------------------------------------------#
# Hansen coefficients.


def hanc(l, m, n, ecc, kmax=10):
    r"""Calculate the Hansen cosine coefficient :math:`C^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\cos\psi` term.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the coefficient.

    Notes
    -----
    The coefficient is defined by

    .. math::
        C^{lm}_n = \int_{t_0}^{t_0+P} (dt/P) \exp(2\pi\mathi nt/P)
        \cos(m\psi) / (1 + e\cos\psi)^l,

    where :math:`P` is the binary period, :math:`e` the eccentricity,
    and :math:`\psi` the true anomaly.
    """

    g = gamma(ecc)
    b = beta(ecc)

    hanc = np.sum(
        [0.5 * b**abs(k) * (1. + b**2.) ** -(l+1.)
         * g ** -(2.*l) / factorial(abs(k))
         * (poch(-m-l-1, max(0,-k)) * poch(m-l-1, max(0,k))
            * jn(n+m+k, n*ecc)
            * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
            + poch(m-l-1, max(0,-k)) * poch(-m-l-1, max(0,k))
            * jn(n-m+k, n*ecc)
            * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.))
         for k in range(-kmax, kmax+1)],
        axis = 0)

    return hanc * (1.+0.j)


def hans(l, m, n, ecc, kmax=10):
    r"""Calculate the Hansen sine coefficient :math:`S^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\sin\psi` term.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the coefficient.

    Notes
    -----
    The coefficient is defined by

    .. math::
        S^{lm}_n = \int_{t_0}^{t_0+P} (dt/P) \exp(2\pi\mathi nt/P)
        \sin(m\psi) / (1 + e\cos\psi)^l,

    where :math:`P` is the binary period, :math:`e` the eccentricity,
    and :math:`\psi` the true anomaly.
    """

    g = gamma(ecc)
    b = beta(ecc)

    hans = np.sum(
        [0.5 * b**abs(k) * (1. + b**2.) ** -(l+1.)
         * g ** -(2.*l) / factorial(abs(k))
         * (poch(-m-l-1, max(0,-k)) * poch(m-l-1, max(0,k))
            * jn(n+m+k, n*ecc)
            * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
            - poch(m-l-1, max(0,-k)) * poch(-m-l-1, max(0,k))
            * jn(n-m+k, n*ecc)
            * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.))
         for k in range(-kmax, kmax+1)],
        axis = 0)

    return hans * 1.j


def hane(l, m, n, ecc, kmax=10):
    r"""Calculate the Hansen coefficient :math:`E^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\cos\psi` and :math:`\sin\psi` terms.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the coefficient.
    """

    return hanc(l, m, n, ecc, kmax) + hans(l, m, n, ecc, kmax)


#-----------------------------------------------------------------------------#
# Derivatives of Hansen coefficients.


def hancp(l, m, n, ecc, kmax=10):
    r"""Calculate the eccentricity derivative of :math:`C^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\cos\psi` term.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the derivative.
    """

    g = gamma(ecc)
    b = beta(ecc)
    gp = gammap(ecc)
    bp = betap(ecc)

    hancp = np.sum(
        [0.25 * b**abs(k) * (1. + b**2.) ** -(l+1.)
         * g ** -(2.*l) / factorial(abs(k))
         * (poch(-m-l-1, max(0,-k)) * poch(m-l-1, max(0,k))
            * (n * (jn(n+m+k-1, n*ecc) - jn(n+m+k+1, n*ecc))
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               - 4. * (l+1.) * b * bp / (1. + b**2.)
               * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 2. * abs(k) * bp / b * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 4. * (-m-l-1.+max(0,-k)) * (m-l-1.+max(0,k))
               * b * bp / (abs(k)+1.) * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l+max(0,-k), m-l+max(0,k), abs(k)+2., b**2.)
               - 4. * l * gp / g * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.))
            + poch(m-l-1, max(0,-k)) * poch(-m-l-1, max(0,k))
            * (n * (jn(n-m+k-1, n*ecc) - jn(n-m+k+1, n*ecc))
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               - 4. * (l+1.) * b * bp / (1. + b**2.)
               * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 2. * abs(k) * bp / b * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 4. * (m-l-1.+max(0,-k)) * (-m-l-1.+max(0,k))
               * b * bp / (abs(k)+1.) * jn(n-m+k, n*ecc)
               * hyp2f1(m-l+max(0,-k), -m-l+max(0,k), abs(k)+2., b**2.)
               - 4. * l * gp / g * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)))
         for k in range(-kmax, kmax+1)],
        axis = 0)

    return hancp * (1.+0.j)


def hansp(l, m, n, ecc, kmax=10):
    r"""Calculate the eccentricity derivative of :math:`S^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\sin\psi` term.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the derivative.
    """

    g = gamma(ecc)
    b = beta(ecc)
    gp = gammap(ecc)
    bp = betap(ecc)

    hansp = np.sum(
        [0.25 * b**abs(k) * (1. + b**2.) ** -(l+1.)
         * g ** -(2.*l) / factorial(abs(k))
         * (poch(-m-l-1, max(0,-k)) * poch(m-l-1, max(0,k))
            * (n * (jn(n+m+k-1, n*ecc) - jn(n+m+k+1, n*ecc))
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               - 4. * (l+1.) * b * bp / (1. + b**2.)
               * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 2. * abs(k) * bp / b * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 4. * (-m-l-1.+max(0,-k)) * (m-l-1.+max(0,k))
               * b * bp / (abs(k)+1.) * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l+max(0,-k), m-l+max(0,k), abs(k)+2., b**2.)
               - 4. * l * gp / g * jn(n+m+k, n*ecc)
               * hyp2f1(-m-l-1.+max(0,-k), m-l-1.+max(0,k), abs(k)+1., b**2.))
            - poch(m-l-1, max(0,-k)) * poch(-m-l-1, max(0,k))
            * (n * (jn(n-m+k-1, n*ecc) - jn(n-m+k+1, n*ecc))
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               - 4. * (l+1.) * b * bp / (1. + b**2.)
               * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 2. * abs(k) * bp / b * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)
               + 4. * (m-l-1.+max(0,-k)) * (-m-l-1.+max(0,k))
               * b * bp / (abs(k)+1.) * jn(n-m+k, n*ecc)
               * hyp2f1(m-l+max(0,-k), -m-l+max(0,k), abs(k)+2., b**2.)
               - 4. * l * gp / g * jn(n-m+k, n*ecc)
               * hyp2f1(m-l-1.+max(0,-k), -m-l-1.+max(0,k), abs(k)+1., b**2.)))
         for k in range(-kmax, kmax+1)],
        axis = 0)

    return hansp * 1.j


def hanep(l, m, n, ecc, kmax=10):
    r"""Calculate the eccentricity derivative of :math:`E^{lm}_n(e)`.

    Parameters
    ----------
    l : int
        Power of the :math:`r/a` term.
    m : int
        Order of the :math:`\cos\psi` and :math:`\sin\psi` terms.
    n : int
        Order of the :math:`\exp(2\pi\mathi nt/P)` term.
    ecc : float or array_like of floats
        Binary eccentricity.
    kmax : int, optional
        Maximum order in the :math:`\beta` expansion.

    Returns
    -------
    complex or array
        Value of the derivative.
    """

    return hancp(l, m, n, ecc, kmax) + hansp(l, m, n, ecc, kmax)


#-----------------------------------------------------------------------------#


def km_funcs(ecc):
    r"""Compute some quantities used in the Kramers-Moyal coefficients.

    Parameters
    ----------
    ecc : float
        Binary eccentricity.

    Returns
    -------
    out : ndarray, shape (10, NMAX)
        Values of the KM coefficient functions.

    Notes
    -----
    An explanation of each of the 10 entries along the first axis is
    given below.  The second axis corresponds to the first NMAX harmonics
    of the binary period.

    out[0] is used for :math:`D^{(1)}_P`
    out[1] is used for :math:`D^{(1)}_e`
    out[2] is used for :math:`D^{(1)}_\omega`, :math:`D^{(2)}_{II}`,
        :math:`D^{(2)}_{I\Omega}`, :math:`D^{(2)}_{I\omega}`,
        and :math:`D^{(2)}_{\Omega\Omega}`
    out[3] is used for :math:`D^{(2)}_{PP}`
    out[4] is used for :math:`D^{(2)}_{Pe}`
    out[5] is used for :math:`D^{(2)}_{ee}`
    out[6] is used for :math:`D^{(2)}_{II}`
        and :math:`D^{(2)}_{\Omega\Omega}`
    out[7] is used for :math:`D^{(2)}_{\omega\omega}`
    out[8] is used for :math:`D^{(2)}_{\omega\epsilon}`
    out[9] is used for :math:`D^{(2)}_{\epsilon\epsilon}`
    """

    nmax = NMAX
    out = np.zeros((10, nmax))
    g = gamma(ecc)

    out[0] = np.array(
        [2.25 * g**2. * n
         * (np.abs(hane(0, 2, n, ecc)
                   + 0.5 * ecc * (hane(1, 1, n, ecc)-hane(1, 3, n, ecc))) ** 2.
            - (1. + 4. * ecc**2.) / 15. * hans(1, 1, n, ecc)**2.
            - ecc * g**2. / 15. * hans(1, 1, n, ecc) * hansp(1, 1, n, ecc)
            + 0.1 * g**2. / ecc
            * ((hane(0, 2, n, ecc)
                + 0.5 * ecc * (hane(1, 1, n, ecc)
                               - hane(1, 3, n, ecc))).conjugate()
               * (3. * hane(1, 1, n, ecc) + hane(1, 3, n, ecc)
                  + ecc * hane(2, 0, n, ecc) + 4. * hane(2, 1, n, ecc)
                  + 4. * ecc * hane(2, 2, n, ecc) - 4. * hane(2, 3, n, ecc)
                  - ecc * hane(2, 4, n, ecc) + 2. * hanep(0, 2, n, ecc)
                  + ecc * (hanep(1, 1, n, ecc)-hanep(1, 3, n, ecc))))
            - 0.1 * g**4. / ecc
            * (hane(2, 2, n, ecc).conjugate()
               * (hane(1, 1, n, ecc) - hane(1, 3, n, ecc)
                  + 2. * hanep(0, 2, n, ecc)
                  + ecc * (hanep(1, 1, n, ecc)-hanep(1, 3, n, ecc))))).real
         for n in range(1, nmax+1)])
    out[1] = np.array(
        [-0.05 * g**6. * n
         * ((3. / ecc**3.
             * (hane(0, 2, n, ecc) - g**2. * hane(2, 2, n, ecc)
                + 0.5 * ecc * (hane(1, 1, n, ecc)
                               - hane(1, 3, n, ecc))).conjugate()
             * (hane(0, 2, n, ecc) - hane(2, 2, n, ecc)
                - ecc * (hane(1, 1, n, ecc) + hane(1, 3, n, ecc)
                         + 2. * hane(2, 1, n, ecc) - 2. * hane(2, 3, n, ecc)
                         + hanep(0, 2, n, ecc) - g**2. * hanep(2, 2, n, ecc))
                - 0.5 * ecc**2.
                * (hane(2, 0, n, ecc) + 10. * hane(2, 2, n, ecc)
                   - hane(2, 4, n, ecc) + hanep(1, 1, n, ecc)
                   - hanep(1, 3, n, ecc))))
            + (hans(1, 1, n, ecc) * hansp(1, 1, n, ecc))).real
         for n in range(1, nmax+1)])
    out[2] = np.array(
        [0.075 * g**6. * n
         * (hane(2, 0, n, ecc) * hane(2, 2, n, ecc).conjugate()).real
         for n in range(1, nmax+1)])
    out[3] = np.array(
        [1.35 * g**2. * n
         * (np.abs(hane(0, 2, n, ecc)
                   + 0.5 * ecc * (hane(1, 1, n, ecc)-hane(1, 3, n, ecc))) ** 2.
            - 1. / 3. * ((ecc*hans(1, 1, n, ecc)) ** 2.).real)
         for n in range(1, nmax+1)])
    out[4] = np.array(
        [-0.225 * g**6. * n
         * (hane(2, 2, n, ecc).conjugate()
            * (2./ecc * hane(0, 2, n, ecc) + hane(1, 1, n, ecc)
               - hane(1, 3, n, ecc))).real
         for n in range(1, nmax+1)])
    out[5] = np.array(
        [0.15 * ecc**-2. * g**6. * n
         * (np.abs(hane(0, 2, n, ecc)
                   + 0.5 * ecc * (hane(1, 1, n, ecc)-hane(1, 3, n, ecc))
                   - g**2. * hane(2, 2, n, ecc)) ** 2.
            - 1./3. * ((ecc*hans(1, 1, n, ecc)) ** 2.).real)
         for n in range(1, nmax+1)])
    out[6] = np.array(
        [0.0375 * g**6. * n * (np.abs(hane(2, 0, n, ecc))**2.
                               + np.abs(hane(2, 2, n, ecc))**2.)
         for n in range(1, nmax+1)])
    out[7] = np.array(
        [0.0375 * ecc**-2. * g**6. * n
         * (np.abs(hane(1, 1, n, ecc) + hane(1, 3, n, ecc)
                   + 2. * (hane(2, 1, n, ecc) - hane(2, 3, n, ecc))
                   + 0.5 * ecc * (hane(2, 0, n, ecc)-hane(2, 4, n, ecc))) ** 2.
            + 4./3. * hanc(1, 1, n, ecc) ** 2.).real
         for n in range(1, nmax+1)])
    out[8] = np.array(
        [-0.0375 * ecc**-2. * g**7. * n
         * (np.abs(hane(1, 1, n, ecc) + hane(1, 3, n, ecc)
                   + 2. * (hane(2, 1, n, ecc)-hane(2, 3, n, ecc))
                   + 0.5 * ecc * (hane(2, 0, n, ecc) - 4.*hane(2, 2, n, ecc)
                                  - hane(2, 4, n, ecc))) ** 2.
            + 4./3. * hanc(1, 1, n, ecc)
            * (hanc(1, 1, n, ecc) - 2.*ecc*hanc(2, 0, n, ecc))
            - 4. * np.abs(ecc*hane(2, 2, n, ecc)) ** 2.).real
         for n in range(1, nmax+1)])
    out[9] = np.array(
        [0.0375 * ecc**-2. * g**8. * n
         * (np.abs(hane(1, 1, n, ecc) + hane(1, 3, n, ecc)
                   + 2. * (hane(2, 1, n, ecc)-hane(2, 3, n, ecc))
                   + 0.5 * ecc * (hane(2, 0, n, ecc) - 8.*hane(2, 2, n, ecc)
                                  - hane(2, 4, n, ecc))) ** 2.
            + 4./3. * np.abs(hanc(1, 1, n, ecc)
                             - 2.*ecc*hanc(2, 0, n, ecc)) ** 2.)
         for n in range(1, nmax+1)])

    return out


# Create a lookup table for the KM coefficients
km_funcs = InterpolatedFunction(km_funcs,
                                np.linspace(ECC_MIN, ECC_MAX, 999))
km_funcs.interpolate()


#-----------------------------------------------------------------------------#


class Binary(object):
    """Class for calculating binary evolution due to SGWB resonance.

    Attributes
    ----------
    sgwb : function
        Present-day SGWB energy density spectrum, as a function of
        frequency (in Hertz).
    t : array of floats
        Times at which the binary's orbital elements are stored.
    x0 : array of floats
        Deterministic values of the orbital elements. First axis
        corresponds to the times in ``t''. Second axis corresponds to
        the different orbital elements.
    dx : array of floats
        Mean stochastic deviations in the orbital elements. First axis
        corresponds to the times in ``t''. Second axis corresponds to
        the different orbital elements.
    cov : array of floats
        Covariance matrix for the orbital elements. First axis
        corresponds to the times in ``t''. Second and third axes
        correspond to the rows and columns of the matrix.
    m1 : float
        Mass of the primary object in solar units.
    m2 : float
        Mass of the secondary object in solar units.
    mtot : float
        Total mass of the binary in solar units.
    eta : float
        Dimensionless mass ratio of the binary.
    nmax : int
        Maximum harmonic of the binary included in the evolution
        equations.
    elements : array of strs
        The names of the binary's orbital elements. Only includes
        elements not initialised as ``None''.
    abbrevs : array of strs
        Abbreviated names of the orbital elements. Only includes
        elements not initialised as ``None''.
    mask : array of bools
        Each entry is ``True'' if the corresponding orbital element
        is defined for this binary, and ``False'' otherwise.
    low_ecc : bool
        If ``False'', use the default orbital elements. If ``True'',
        replace the eccentricity and argument of pericentre with the
        Laplace-Lagrange parameters, replace the compensated mean
        anomaly with the compensated mean longitude, and neglect terms
        of order eccentricity squared in the evolution equations.
    lookback : float
        Cosmological lookback time to the moment when the binary is
        initialised, in seconds. If ``lookback'' is greater than zero,
        then the variation of the SGWB enery density over cosmological
        timescales is included in the orbital evolution, with ``sgwb''
        taken as the present-day density. If ``lookback'' is zero
        (default), then the SGWB is taken to be constant in time.

    Methods
    -------
    evolve_fokker_planck(t_stop, t_eval)
        Evolve the binary's distribution function forward in time.
    evolve_langevin(t_stop, t_eval, n_walks)
        Evolve random realisations of the binary forward in time.
    v_sec(x)
        Calculate the deterministic drift for the orbital elements.
    dv_sec(x)
        First derivatives of the deterministic drift.
    ddv_sec(x)
        Second derivatives of the deterministic drift.
    km1(x, t)
        Calculate the stochastic drift due to the SGWB.
    km2(x, t)
        Calculate the stochastic diffusion due to the SGWB.
    plot(...)
        Plot the orbital elements.
    """

    def __init__(self,
                 sgwb,
                 x_init,
                 m1=1.35,
                 m2=1.35,
                 low_ecc=False,
                 lookback=0.,
                 ):
        """Initialise the Binary instance.

        Parameters
        ----------
        sgwb : float or function
            The (present-day) SGWB energy density spectrum, as a
            function of frequency (in Hertz). If ``sgwb'' is a float,
            the spectrum is assumed to be scale-invariant.
        x_init : float or array_like of shape (6,)
            Initial orbital elements of the binary. See Notes below for
            details.
        m1 : float, optional
            Mass of the primary object in solar units.
        m2 : float, optional
            Mass of the secondary object in solar units.
        low_ecc : bool, optional
            If ``False'', use the default orbital elements. If ``True'',
            replace the eccentricity and argument of pericentre with the
            Laplace-Lagrange parameters, replace the compensated mean
            anomaly with the compensated mean longitude, and neglect terms
            of order eccentricity squared in the evolution equations.
        lookback : float, optional
            Cosmological lookback time to the moment when the binary is
            initialised, in seconds. If ``lookback'' is greater than zero,
            then the redshifting of the SGWB enery density and GW
            frequencies over cosmological timescales is included in the
            orbital evolution, with ``sgwb'' taken as the present-day
            spectrum. If ``lookback'' is zero (default), then the SGWB
            is taken to be constant in time.

        Notes
        -----
        There are at most six orbital elements. In the standard case (where
        ``low_ecc'' is False) these are:
            0. period (``per''),
            1. eccentricity (``ecc''),
            2. inclination (``inc''),
            3. longitude of ascending node (``asc''),
            4. argument of pericentre (``arg''),
            5. compensated mean anomaly (``eps'').

        The standard orbital elements are always listed in the order above.
        For example, ``x0[i,2]'' returns the deterministic inclination of
        the binary at the i-th timestep, and ``cov[j,0,4]'' returns the
        covariance of the period with the argument of pericentre at the
        j-th timestep. The orbital element arrays that are inputted into
        the class methods such as ``v_sec'' and ``km1'', etc., are assumed
        to follow the same ordering.

        If ``low_ecc'' is True, then the eccentricity and argument of
        pericentre are replaced by the first and second Laplace-Lagrange
        parameters (``zet'' and ``kap''), and the compensated mean anomaly
        is replaced by the compensated mean longitude (``xxi''). These
        alternative orbital elements are always listed in the following
        order:
            0. period (``per''),
            1. 1st Laplace-Lagrange parameter (``zet''),
            2. 2nd Laplace-Lagrange parameter (``kap''),
            3. inclination (``inc''),
            4. longitude of ascending node (``asc''),
            5. compensated mean longitude (``xxi'').

        When initialising the class instance, one can specify just a subset
        of the oribtal elements by entering ``None'' for the unwanted
        elements. For example, to specify just the period (e.g. one day)
        and eccentricity (e.g. 0.5) but nothing else, one would set
            ``x_init = [DAY, 0.5, None, None, None, None]''.
        In the simplest case where one is only interested in the period of
        the binary, one can input this as a float rather than an array,
            ``x_init = DAY'',
        which is equivalent to
            ``x_init = [DAY, None, None, None, None, None]''.

        Note that there are some subsets of orbital elements that are
        invalid. For example, it is not possible to specify the argument of
        pericentre without also specifying the eccentricity. Entering an
        invalid set of orbital elements will result in a ValueError.
        """

        if type(sgwb) is float:
            const = sgwb
            sgwb = lambda freq: const

        if not isfunction(sgwb):
            raise TypeError("'sgwb' must be a function or a float.")

        if type(x_init) is float:
            x_init = [x_init] + 5*[None]

        x_init = np.array(x_init)

        if x_init.shape != (6,):
            raise ValueError("Initial orbital elements must be a float or an "
                             "array of shape (6,).")

        if not low_ecc:
            self.elements = np.array(["Period",
                                      "Eccentricity",
                                      "Inclination",
                                      "Longitude of ascending node",
                                      "Argument of pericentre",
                                      "Compensated mean anomaly"])
            self.abbrevs = np.array(["per", "ecc", "inc", "asc", "arg", "eps"])
            (per, ecc, inc, asc, arg, eps) = x_init

            if per is None:
                raise ValueError("The binary period must be specified.")

            if (asc is not None) and (inc is None):
                raise ValueError("Longitude of ascending node cannot be "
                                 "specified without inclination.")

            if (eps is not None) and (inc is None):
                raise ValueError("Compensated mean anomaly cannot be specified"
                                 " without inclination.")

            if (arg is not None) and (ecc is None):
                raise ValueError("Argument of pericentre cannot be specified "
                                 "without eccentricity.")

            if ecc is not None:
                if (inc is not None) and (arg is None):
                    raise ValueError(
                        "Inclination cannot be specified without argument of "
                        "pericentre for an eccentric binary.")

                if (asc is not None) and (arg is None):
                    raise ValueError("Longitude of ascending node cannot be "
                                     "specified without argument of pericentre"
                                     " for an eccentric binary.")

                if (arg is not None) and (inc is None):
                    raise ValueError("Argument of pericentre cannot be "
                                     "specified without inclination for an "
                                     "eccentric binary.")

                if (eps is not None) and (inc is None):
                    raise ValueError("Compensated mean anomaly cannot be "
                                     "specified without inclination for an "
                                     "eccentric binary.")

                if (eps is not None) and (arg is None):
                    raise ValueError("Compensated mean anomaly cannot be "
                                     "specified without argument of pericentre"
                                     " for an eccentric binary.")

        if low_ecc:
            self.elements = np.array(["Period",
                                      "First Laplace-Lagrange parameter",
                                      "Second Laplace-Lagrange parameter",
                                      "Inclination",
                                      "Longitude of ascending node",
                                      "Compensated mean longitude"])
            self.abbrevs = np.array(["per", "zet", "kap", "inc", "asc", "xxi"])
            (per, zet, kap, inc, asc, xxi) = x_init

            if per is None:
                raise ValueError("You must specify the binary period.")

            if (((zet is not None) and (kap is None))
                or ((kap is not None) and (zet is None))):
                raise ValueError("You cannot specify one Laplace-Lagrange "
                                 "parameter without specifying the other.")

            if (asc is not None) and (inc is None):
                raise ValueError("Longitude of ascending node cannot be "
                                 "specified without inclination.")

            if (xxi is not None) and (inc is None):
                raise ValueError("Compensated mean longitude cannot be "
                                 "specified without inclination.")

        self.mask = (x_init != None)
        self._y_mask = np.array(2 * list(self.mask)
                               + list(np.outer(self.mask,
                                               self.mask)[np.triu_indices(6)]))
        x_init[x_init == None] = 0.

        self.t = [0.]
        self._y = np.array(list(x_init) + [0.]*27)

        self.elements = self.elements[self.mask]
        self.abbrevs = self.abbrevs[self.mask]
        self._n_el = self.elements.size

        self.x0 = np.array([x_init[self.mask]],
                            dtype=float)
        self.dx = np.array([np.zeros(self._n_el)],
                            dtype=float)
        self.cov = np.array([np.zeros((self._n_el, self._n_el))],
                            dtype=float)

        self.m1 = m1
        self.m2 = m2
        self.mtot = m1 + m2
        self.eta = m1 * m2 * self.mtot**-2.

        self.low_ecc = low_ecc
        self.sgwb = sgwb
        self.lookback = lookback
        self.nmax = min(int(np.floor(v_rms(per, self.mtot) ** -1.)), NMAX)

        self.walks = None

    def v_sec(self, x):
        """Calculate the deterministic drift for the orbital elements.

        Parameters
        ----------
        x : array_like
            Values of the orbital elements.

        Returns
        -------
        out : array
            Values of the deterministic drift.
        """

        if np.array(x).ndim == 0:
            x = [x]

        short_output = False

        if len(x) < 6:
            short_output = True
            xx = np.zeros(6)
            xx[self.mask] = x
            x = xx

        out = np.zeros(6)

        if not self.low_ecc:
            (per, ecc, inc, asc, arg, eps) = x
            v = v_rms(per, self.mtot)
            g = gamma(ecc)

            out[0] = (-192. / 5. * np.pi * self.eta * v**5. * g**-7.
                      * (1. + 73. / 24. * ecc**2. + 37. / 96. * ecc**4.))

            if self.mask[1]:
                out[1] = (-608. / 15. * np.pi * self.eta * v**5. * g**-5. / per
                          * (ecc + 121. / 304. * ecc**3.))

            if self.mask[4]:
                out[4] = 6. * np.pi * v**2. * g**-2. / per

            if self.mask[5]:
                out[5] = (2. * np.pi * v**2. / per
                          * (6. - 7. * self.eta - (15. - 9. * self.eta) / g))

        if self.low_ecc:
            (per, zet, kap, inc, asc, xxi) = x
            v = v_rms(per, self.mtot)

            out[0] = -192. / 5. * np.pi * self.eta * v**5.

            if self.mask[1] and self.mask[2]:
                out[1] = (6. * np.pi * v**2. / per
                          * (kap - 304. / 45. * self.eta * v**3. * zet))
                out[2] = -(6. * np.pi * v**2. / per
                           * (zet + 304. / 45. * self.eta * v**3. * kap))

            if self.mask[5]:
                out[5] = -4. * np.pi * v**2. / per * (3.-self.eta)

        if short_output:
            out = out[self.mask]

            while out.ndim < 1:
                out = np.array([out])

        return out

    def dv_sec(self, x):
        """First derivatives of the deterministic drift.

        Parameters
        ----------
        x : array_like
            Values of the orbital elements.

        Returns
        -------
        out : array
            Values of the derivatives. The first axis correspond to the
            index on the partial derivative, while the second axis
            corresponds to the index on the deterministic drift vector.
        """

        if np.array(x).ndim == 0:
            x = [x]

        short_output = False

        if len(x) < 6:
            short_output = True
            xx = np.zeros(6)
            xx[self.mask] = x
            x = xx

        out = np.zeros((6, 6))

        if not self.low_ecc:
            (per, ecc, inc, asc, arg, eps) = x
            v = v_rms(per, self.mtot)
            g = gamma(ecc)

            out[0, 0] = (64. * np.pi * self.eta * v**5. * g**-7. / per
                         * (1. + 73. / 24. * ecc**2. + 37. / 96. * ecc**4.))

            if self.mask[1]:
                out[1, 0] = (-2512. / 5. * np.pi * self.eta * v**5. * g**-9.
                             * (ecc + 201. / 157. * ecc**3.
                                + 111. / 1256. * ecc**5.))
                out[0, 1] = (4864. / 45. * np.pi * self.eta
                             * v**5. * g**-5. / per**2.
                             * (ecc + 121. / 304. * ecc**3.))
                out[1, 1] = (-608. / 15. * np.pi * self.eta * v**5. * g**-7.
                             / per * (1. + 1579. / 304. * ecc**2.
                                      + 121. / 152. * ecc**4.))

            if self.mask[4]:
                out[0, 4] = -10. * np.pi * v**2. * g**-2. / per**2.

            if self.mask[1] and self.mask[4]:
                out[1, 4] = 12. * np.pi * ecc * v**2. * g**-4. / per

            if self.mask[5]:
                out[0, 5] = (-10. / 3. * np.pi * v**2. / per**2.
                             * (6. - 7.*self.eta - (15. - 9.*self.eta) / g))

            if self.mask[1] and self.mask[5]:
                out[1, 5] = (-6. * np.pi * ecc * v**2. * g**-3. / per
                             * (5. - 3.*self.eta))

        if self.low_ecc:
            (per, zet, kap, inc, asc, xxi) = x
            v = v_rms(per, self.mtot)

            out[0, 0] = 64. * np.pi * self.eta * v**5. / per

            if self.mask[1] and self.mask[2]:
                out[0, 1] = -(10. * np.pi * (v/per) ** 2.
                              * (kap - 2432. / 225. * self.eta * v**3. * zet))
                out[0, 2] = (10. * np.pi * (v/per) ** 2.
                             * (zet + 2432. / 225. * self.eta * v**3. * kap))
                out[1, 1] = -608. / 15. * np.pi * self.eta * v**5. / per**2.
                out[1, 2] = 6. * np.pi * v**2. / per
                out[2, 1] = -6. * np.pi * v**2. / per
                out[2, 2] = -608. / 15. * np.pi * self.eta * v**5. / per**2.

            if self.mask[5]:
                out[0, 5] = 20. / 3. * np.pi * (v/per) ** 2. * (3.-self.eta)

        if short_output:
            out = out[self.mask, self.mask]

            while out.ndim < 2:
                out = np.array([out])

        return out

    def ddv_sec(self, x):
        """Second derivatives of the deterministic drift.

        Parameters
        ----------
        x : array_like
            Values of the orbital elements.

        Returns
        -------
        out : array
            Values of the derivatives. The first and second axes
            correspond to the indices on the two partial derivatives,
            while the third axis corresponds to the index on the
            deterministic drift vector.
        """

        if np.array(x).ndim == 0:
            x = [x]

        short_output = False

        if len(x) < 6:
            short_output = True
            xx = np.zeros(6)
            xx[self.mask] = x
            x = xx

        out = np.zeros((6, 6, 6))

        if not self.low_ecc:
            (per, ecc, inc, asc, arg, eps) = x
            v = v_rms(per, self.mtot)
            g = gamma(ecc)

            out[0, 0, 0] = (-512. / 3. * np.pi * self.eta
                            * v**5. * g**-7. / per**2.
                            * (1. + 73. / 24. * ecc**2. + 37. / 96. * ecc**4.))

            if self.mask[1]:
                out[0, 1, 0] = (2512. / 3. * np.pi * self.eta
                                * v**5. * g**-9. / per
                                * (ecc + 201. / 157. * ecc**3.
                                   + 111. / 1256. * ecc**5.))
                out[1, 0, 0] = out[0, 1, 0]
                out[1, 1, 0] = (-502.4 * np.pi * self.eta * v**5. * g**-11.
                                * (1. + 1859. / 157. * ecc**2.
                                   + 10203. / 1256. * ecc**4.
                                   + 111. / 314. * ecc**6.))
                out[0, 0, 1] = (-53504. / 135. * np.pi * self.eta
                                * v**5. * g**-5. / per**3.
                                * (ecc + 121. / 304. * ecc**3.))
                out[0, 1, 1] = (4864. / 45. * np.pi * self.eta
                                * v**5. * g**-7. / per**2.
                                * (1. + 1579. / 304. * ecc**2.
                                   + 121. / 152. * ecc**4.))
                out[1, 0, 1] = out[0, 1, 1]
                out[1, 1, 1] = (-704.8 * np.pi * self.eta
                                * v**5. * g**-9. / per
                                * (ecc + 8863. / 5286. * ecc**3.
                                   + 121. / 881. * ecc**5.))

            if self.mask[4]:
                out[0, 0, 4] = 80. / 3. * np.pi * v**2. * g**-2. / per**3.

            if self.mask[1] and self.mask[4]:
                out[0, 1, 4] = 20. * np.pi * ecc * v**2. * g**-4. / per**2.
                out[1, 0, 4] = out[0, 1, 4]
                out[1, 1, 4] = (12. * np.pi * v**2. * g**-3.
                                * (1. + 3. * ecc**2.) / per)

            if self.mask[5]:
                out[0, 0, 5] = (80. / 9. * np.pi * v**2. / per**3.
                                * (6. - 7.*self.eta - (15. - 9.*self.eta) / g))

            if self.mask[1] and self.mask[5]:
                out[0, 1, 5] = (10. * np.pi * ecc * v**2. * g**-3. / per**2.
                                * (5. - 3.*self.eta))
                out[1, 0, 5] = out[0, 1, 5]
                out[1, 1, 5] = (-6. * np.pi * v**2. * g**-5. / per
                                * (1. + 2. * ecc**2.) * (5. - 3.*self.eta))

        if self.low_ecc:
            (per, zet, kap, inc, asc, xxi) = x
            v = v_rms(per, self.mtot)

            out[0, 0, 0] = -512. / 3. * np.pi * self.eta * v**5. / per**2.

            if self.mask[1] and self.mask[2]:
                out[0, 0, 1] = (80. / 3. * np.pi * v**2. / per**3.
                                * (kap
                                   - 3344. / 225. * self.eta * v**3. * zet))
                out[0, 1, 1] = 4864. / 45. * np.pi * self.eta * v**5. / per**2.
                out[0, 2, 1] = -10. * np.pi * (v/per) ** 2.
                out[1, 0, 1] = out[0, 1, 1]
                out[2, 0, 1] = out[0, 2, 1]
                out[0, 0, 2] = -(80. / 3. * np.pi * v**2. / per**3.
                                 * (zet
                                    + 3344. / 225. * self.eta * v**3. * kap))
                out[0, 1, 2] = 10. * np.pi * (v/per) ** 2.
                out[0, 2, 2] = 4864. / 45. * np.pi * self.eta * v**5. / per**2.
                out[1, 0, 2] = out[0, 1, 2]
                out[2, 0, 2] = out[0, 2, 2]

            if self.mask[5]:
                out[0, 0, 5] = -160. / 9. * np.pi / per**3. * (3.-self.eta)

        if short_output:
            out = out[self.mask, self.mask, self.mask]

            while out.ndim < 3:
                out = np.array([out])

        return out

    def km1(self, x, t=0.):
        """Calculate the stochastic drift due to the SGWB.

        Parameters
        ----------
        x : array_like
            Values of the orbital elements.
        t : float, optional
            Time, in seconds, measured from the moment at which the
            binary is initialised. Note that this only affects the
            output if ``lookback'' is nonzero.

        Returns
        -------
        out : array
            Values of the stochastic drift.

        Notes
        -----
        Despite the name of this function, this is not strictly the
        first Kramers-Moyal coefficient, just the stochastic part
        thereof. The whole coefficient is given by adding the
        deterministic part ``v_sec''.
        """

        if np.array(x).ndim == 0:
            x = [x]

        short_output = False

        if len(x) < 6:
            short_output = True
            xx = np.zeros(6)
            xx[self.mask] = x
            x = xx

        out = np.zeros(6)

        if not self.low_ecc:
            (per, ecc, inc, asc, arg, eps) = x

            if (not self.mask[1]) or (ecc < ECC_MIN):

                if self.lookback > 0.:
                    z = redshift(t)
                    gw1 = (1.+z)**4. * self.sgwb((1.+z)/per)
                    gw2 = (1.+z)**4. * self.sgwb(2.*(1.+z)/per)
                    gw3 = (1.+z)**4. * self.sgwb(3.*(1.+z)/per)

                else:
                    gw1 = self.sgwb(1./per)
                    gw2 = self.sgwb(2./per)
                    gw3 = self.sgwb(3./per)

                out[0] = (3. / 160. * (per*HUB0) ** 2.
                          * (288.*gw2 - 79.*gw1 - 27.*gw3))

                if self.mask[2]:
                    out[2] = 3. / 80. * per * HUB0**2. * gw2 / np.tan(inc)

            else:
                km = km_funcs(ecc)
                g = gamma(ecc)

                if self.lookback > 0.:
                    z = redshift(t)
                    gw = np.array([(1.+z)**4. * self.sgwb(n*(1.+z)/per)
                                   for n in range(1, self.nmax+1)])

                else:
                    gw = np.array([self.sgwb(n/per)
                                   for n in range(1, self.nmax+1)])

                out[0] = np.dot(km[0, :self.nmax], (per*HUB0) ** 2. * gw)
                out[1] = np.dot(km[1, :self.nmax], per * HUB0**2. * gw)

                if self.mask[2]:
                    out[2] =  np.dot(km[6, :self.nmax],
                                     per * HUB0**2. / np.tan(inc) * gw)
                    out[2] -= np.dot(km[2, :self.nmax],
                                     per * HUB0**2. * np.cos(2.*arg)
                                     / np.tan(inc) * gw)

                if self.mask[3]:
                    out[3] = np.dot(km[2, :self.nmax],
                                    -2. * per * HUB0**2. * np.sin(2.*arg)
                                    * np.cos(inc) / np.sin(inc)**2. * gw)

                if self.mask[4]:
                    out[4] = np.dot(km[2, :self.nmax],
                                    per * HUB0**2. * (2. - np.sin(inc)**2.)
                                    * np.sin(2.*arg) / np.sin(inc)**2. * gw)

        if self.low_ecc:
            (per, zet, kap, inc, asc, xxi) = x

            if self.lookback > 0.:
                z = redshift(t)
                gw1 = (1.+z)**4. * self.sgwb((1.+z)/per)
                gw2 = (1.+z)**4. * self.sgwb(2.*(1.+z)/per)
                gw3 = (1.+z)**4. * self.sgwb(3.*(1.+z)/per)

            else:
                gw1 = self.sgwb(1./per)
                gw2 = self.sgwb(2./per)
                gw3 = self.sgwb(3./per)

            out[0] = (3. / 160. * (per*HUB0) ** 2.
                      * (288.*gw2 - 79.*gw1 - 27.*gw3))

            if self.mask[1] and self.mask[2]:
                out[1] = (HUB0**2. * per / 160. * zet / (zet**2. + kap**2.)
                          * (25.*gw1 - 27.*gw3))
                out[2] = (HUB0**2. * per / 160. * kap / (zet**2. + kap**2.)
                          * (25.*gw1 - 27.*gw3))

            if self.mask[3]:
                out[3] = 3. / 80. * per * HUB0**2. * gw2 / np.tan(inc)**2.

        if short_output:
            out = out[self.mask]

            while out.ndim < 1:
                out = np.array([out])

        return out

    def km2(self, x, t=0.):
        """Calculate the stochastic diffusion due to the SGWB.

        Parameters
        ----------
        x : array_like
            Values of the orbital elements.
        t : float, optional
            Time, in seconds, measured from the moment at which the
            binary is initialised. Note that this only affects the
            output if ``lookback'' is nonzero.

        Returns
        -------
        out : array
            Values of the stochastic diffusion.
        """

        if np.array(x).ndim == 0:
            x = [x]

        short_output = False

        if len(x) < 6:
            short_output = True
            xx = np.zeros(6)
            xx[self.mask] = x
            x = xx

        out = np.zeros((6, 6))

        if not self.low_ecc:
            (per, ecc, inc, asc, arg, eps) = x

            if (not self.mask[1]) or (ecc < 1.e-3):

                if self.lookback > 0.:
                    z = redshift(t)
                    gw2 = (1.+z)**4. * self.sgwb(2.*(1.+z)/per)

                else:
                    gw2 = self.sgwb(2./per)

                out[0, 0] = 27. / 20. * per**3. * HUB0**2. * gw2

                if self.mask[2]:
                    out[2, 2] = 3. / 80. * per * HUB0**2. * gw2

                if self.mask[3]:
                    out[3, 3] = out[2, 2] / np.sin(inc)**2.

                if self.mask[5]:
                    out[5, 5] = out[2, 2] * (15. + 2./(1.+np.cos(inc)))

                if self.mask[3] and self.mask[5]:
                    out[3, 5] = out[2, 2] / (1.+np.cos(inc))
                    out[5, 3] = out[3, 5]

            else:
                g  = gamma(ecc)
                km = km_funcs(ecc)

                if self.lookback > 0.:
                    z = redshift(t)
                    gw = np.array([(1.+z)**4. * self.sgwb(n*(1.+z)/per)
                                   for n in range(1, self.nmax+1)])

                else:
                    gw = np.array([self.sgwb(n/per)
                                   for n in range(1, self.nmax+1)])

                out[0, 0] = np.dot(km[3, :self.nmax], per**3. * HUB0**2. * gw)
                out[0, 1] = np.dot(km[4, :self.nmax], (per*HUB0) ** 2. * gw)
                out[0, 1] += out[0, 0] * g**2. / (3.*per*ecc)
                out[1, 0] = out[0, 1]
                out[1, 1] = np.dot(km[5, :self.nmax], per * HUB0**2. * gw)

                if self.mask[2]:
                    out[2, 2] = np.dot(km[6, :self.nmax], per * HUB0**2. * gw)
                    out[2, 2] += np.dot(km[2, :self.nmax],
                                        per * HUB0**2. * np.cos(2.*arg) * gw)

                if self.mask[3]:
                    out[3, 3] = np.dot(km[6, :self.nmax],
                                       per * HUB0**2. / np.sin(inc)**2. * gw)
                    out[3, 3] -= np.dot(km[2, :self.nmax],
                                        per * HUB0**2. * np.cos(2.*arg)
                                        / np.sin(inc)**2. * gw)

                if self.mask[4]:
                    out[4, 4] = np.dot(km[7, :self.nmax], per * HUB0**2. * gw)
                    out[4, 4] += np.dot(km[6, :self.nmax],
                                        per * HUB0**2. / np.tan(inc)**2. * gw)
                    out[4, 4] -= np.dot(km[2, :self.nmax],
                                        per * HUB0**2. * np.cos(2.*arg)
                                        / np.tan(inc)**2. * gw)

                if self.mask[5]:
                    out[5, 5] = np.dot(km[9, :self.nmax], per * HUB0**2. * gw)

                if self.mask[2] and self.mask[3]:
                    out[2, 3] = np.dot(km[2, :self.nmax],
                                       per * HUB0**2. * np.sin(2.*arg)
                                       / np.sin(inc) * gw)
                    out[3, 2] = out[2, 3]

                if self.mask[2] and self.mask[4]:
                    out[2, 4] = np.dot(km[2, :self.nmax],
                                       -per * HUB0**2. * np.sin(2.*arg)
                                       / np.tan(inc) * gw)
                    out[4, 2] = out[2, 4]

                if self.mask[3] and self.mask[4]:
                    out[3, 4] = -np.cos(inc) * out[3, 3]
                    out[4, 3] = out[3, 4]

                if self.mask[4] and self.mask[5]:
                    out[4, 5] = np.dot(km[8, :self.nmax], per * HUB0**2. * gw)
                    out[5, 4] = out[4, 5]

        if self.low_ecc:
            (per, zet, kap, inc, asc, xxi) = x

            if self.lookback > 0.:
                z = redshift(t)
                gw1 = (1.+z)**4. * self.sgwb((1.+z)/per)
                gw2 = (1.+z)**4. * self.sgwb(2.*(1.+z)/per)
                gw3 = (1.+z)**4. * self.sgwb(3.*(1.+z)/per)

            else:
                gw1 = self.sgwb(1./per)
                gw2 = self.sgwb(2./per)
                gw3 = self.sgwb(3./per)

            out[0, 0] = 27. / 20. * per**3. * HUB0**2. * gw2

            if self.mask[1] and self.mask[2]:
                out[0, 1] = (-3. / 160. * per * zet * HUB0**2.
                             * (25.*gw1 + 12.*gw2 - 27.*gw3))
                out[1, 0] = out[0, 1]
                out[0, 2] = (-3. / 160. * per * kap * HUB0**2.
                             * (25.*gw1 + 12.*gw2 - 27.*gw3))
                out[2, 0] = out[0, 2]
                out[1, 1] = per / 160. * HUB0**2. * (29.*gw1 + 9.*gw3)
                out[2, 2] = out[1, 1]

            if self.mask[3]:
                out[3, 3] = 3. / 80. * per * HUB0**2. * gw2

            if self.mask[4]:
                out[4, 4] = out[3, 3] / np.sin(inc)**2.

            if self.mask[5]:
                out[5, 5] = (3. / 80. * per * HUB0**2. * gw2
                             * (16. + np.tan(inc)**-2.))

            if self.mask[1] and self.mask[2] and self.mask[4]:
                out[1, 4] = (-3. / 80. * per * kap * HUB0**2. * gw2
                             * np.cos(inc) / np.sin(inc)**2.)
                out[4, 1] = out[1, 4]
                out[2, 4] = (3. / 80. * per * zet * HUB0**2. * gw2
                             * np.cos(inc) / np.sin(inc)**2.)
                out[4, 2] = out[2, 4]

            if self.mask[1] and self.mask[2] and self.mask[5]:
                out[1, 5] = (-per * kap / 320. * HUB0**2.
                             * (203.*gw1 - 12.*gw2*(20.+np.tan(inc)**-2.)
                                + 63.*gw3))
                out[5, 1] = out[1, 5]
                out[2, 5] = (per * zet / 320. * HUB0**2.
                             * (203.*gw1 - 12.*gw2*(20.+np.tan(inc)**-2.)
                                + 63.*gw3))
                out[5, 2] = out[2, 5]

            if self.mask[4] and self.mask[5]:
                out[4, 5] = (-3. / 80. * per * HUB0**2. * gw2
                             * np.cos(inc) / np.sin(inc)**2.)
                out[5, 4] = out[4, 5]

        if short_output:
            out = out[self.mask, self.mask]

            while out.ndim < 2:
                out = np.array([out])

        return out

    def _dydt(self, t, y):
        """Internal function that defines the initial value problem.

        Parameters
        ----------
        t : float
            Time in seconds.
        y : array, shape (33, ) or (33, n)
            State vector of the system. First six elements are the
            deterministic orbital elements ``x0'', next six elements
            are the mean stochastic corrections ``dx'', and the
            remaining elements are the upper triangle of the covariance
            matrix ``cov'' (the lower triangle is redundant and is
            excluded for efficiency). If shape is ``(33, n)'', this
            represents ``n'' different binaries.

        Returns
        -------
        array, shape (33, ) or (33, n)
            Time derivative of the state vector.
        """

        if len(y.shape) == 1:
            y = np.transpose([y])

        n = y.shape[-1]
        x0 = np.array(y[:6])
        dx = np.array(y[6:12])
        cov = np.zeros((6, 6, n))
        cov[np.triu_indices(6)] = y[12:]
        cov += np.transpose(np.tril(np.transpose(cov), -1), axes=[1, 2, 0])

        dydt = np.zeros_like(y)
        dydt[:6] = np.transpose([self.v_sec(x0[:, i]) for i in range(n)])
        dydt[6:12] = np.transpose([self.km1(x0[:, i], t=t)
                                   + dx[:, i].dot(self.dv_sec(x0[:, i]))
                                   + 0.5*np.tensordot(cov[:, :, i],
                                                      self.ddv_sec(x0[:, i]))
                                   for i in range(n)])
        dydt[12:] = np.transpose([(
            2.*self.km2(x0[:, 0], t=t) + cov[:, :, 0].dot(self.dv_sec(x0[:, 0]))
            + np.transpose(cov[:, :, 0].dot(self.dv_sec(x0[:, 0])))
            )[np.triu_indices(6)]
            for i in range(n)])
        
        if n == 1:
            dydt = dydt[:, 0]

        return dydt
    
    def _rk4_vec(self, t, y, h):
        """Internal function that evolves a vector of binaries
        forwards using RK4.

        Parameters
        ----------
        t : float
            Time in seconds.
        y : array, shape (33, ) or (33, n)
            State vector of the system. First six elements are the
            deterministic orbital elements ``x0'', next six elements
            are the mean stochastic corrections ``dx'', and the
            remaining elements are the upper triangle of the covariance
            matrix ``cov'' (the lower triangle is redundant and is
            excluded for efficiency). If shape is ``(33, n)'', this
            represents ``n'' different binaries.
        h : float
            Size of the forwards time step, in seconds.

        Returns
        -------
        array, shape (33, ) or (33, n)
            State vector, evolved forward by time ``h''.
        """

        out = y

        k = self._dydt(t, y)
        out += h / 6. * k

        k = self._dydt(t + 0.5*h, y + 0.5*h*k)
        out += h / 3. * k

        k = self._dydt(t + 0.5*h, y + 0.5*h*k)
        out += h / 3. * k

        k = self._dydt(t + h, y + h*k)
        out += h / 6. * k

        return out

    def evolve_fokker_planck(self,
                             t_stop,
                             t_eval=None,
                             ):
        """Evolve the binary's distribution function forward in time.

        Solves the coupled set of ODEs governing the secular evolution
        of the mean vector and covariance matrix of the orbital
        elements, to first order in the slow-diffusion approximation.
        The resulting orbital elements are appended to the ``x0'',
        ``dx'', and ``cov'' attributes of the class instance.

        !! Calling this will erase any previous time evolution !!

        Parameters
        ----------
        t_stop : float
            Time at which to stop the evolution, in seconds.
        t_eval : array_like, optional
            Array of times at which the values of the orbital elements
            should be calculated.  Should consist of floats, starting
            with ``0.'' and ending with ``t_stop''.
        """

        if self.t[-1] > 0.:
            self.__init__(self.sgwb,
                          self.x0[0],
                          self.m1,
                          self.m2,
                          self.low_ecc,
                          self.lookback,
                          )

        if self.lookback > 0. and t_stop > self.lookback:
            raise ValueError("Cannot evolve the binary forward to negative"
                             "lookback times. The evolution must end at the"
                             "present day.")

        if t_eval is None:
            t_eval = np.linspace(self.t[-1],
                                 self.t[-1] + t_stop,
                                 1000)

        sol = solve_ivp(self._dydt,
                        [self.t[-1],
                         self.t[-1] + t_stop],
                        self._y,
                        t_eval=t_eval,
                        vectorized=True,
                        method='DOP853')

        self._y = sol.y[:, -1]

        if sol.t[0] == self.t[-1]:
            t = sol.t[1:]
            y = sol.y[self._y_mask, 1:]

        else:
            t = sol.t
            y = sol.y[self._y_mask, :]

        self.t  = np.array(list(self.t) + list(t),
                           dtype=float)
        self.x0 = np.array(list(self.x0)
                           + list(y[:self._n_el].transpose()),
                           dtype=float)
        self.dx = np.array(list(self.dx)
                           + list(y[self._n_el:2*self._n_el].transpose()),
                           dtype=float)

        C = np.zeros((self._n_el,
                      self._n_el,
                      len(sol.t) - 1))
        C[np.triu_indices(self._n_el)[0],
          np.triu_indices(self._n_el)[1]] = y[2*self._n_el:]
        C += C.transpose(1, 0, 2)
        C[np.diag_indices(self._n_el)[0],
          np.diag_indices(self._n_el)[1]] *= 0.5
        self.cov = np.array(list(self.cov) + list(C.transpose()),
                            dtype=float)

        return None

    def evolve_langevin(self,
                        t_stop,
                        t_eval=None,
                        n_walks=1,
                        ):
        """Evolve some number of random trajectories forward in time.

        Solves ``n_walks'' statistically independent realisations of
        the Langevin equation for the binary. Each step is a draw from
        the transition probability, found by solving the Fokker-Planck
        equation for that interval. The resulting random walks are
        saved in the ``walks'' attribute of the class instance, while
        the corresponding time and mean orbital elements are saved in
        ``t'' and ``x0''.

        !! Calling this will erase any previous time evolution !!

        Parameters
        ----------
        n_walks : int
            Number of random walks to evolve.
        t_stop : float
            Time after the present at which to stop the evolution, in
            seconds.
        t_eval : array_like, optional
            Array of times at which the values of the orbital elements
            should be calculated.  Should consist of floats with values
            between ``self.t[-1]'' and ``self.t[-1] + t_stop''.
        """

        if self.t[-1] > 0.:
            self.__init__(self.sgwb,
                          self.x0[0],
                          self.m1,
                          self.m2,
                          self.low_ecc,
                          self.lookback,
                          )

        if t_eval is None:
            t_eval = np.linspace(0., t_stop, 1000)

        nt = len(t_eval)
        self.walks = np.zeros((nt, 6, n_walks))

        for i in tqdm(range(nt-1),
                      desc='Evolving random walks'):
            y = np.zeros((33, n_walks))
            y[:6] = np.outer(self.x0[i], np.ones(n_walks)) + self.walks[i]
            y = self._rk4_vec(t_eval[i], y, t_eval[i+1]-t_eval[i])
            self.t = np.append(self.t, [t_eval[i+1]])
            self.x0 = np.append(self.x0, [np.average(y[:6], axis=-1)], axis=0)
            dx = y[6:12]
            cov = np.zeros((6, 6, n_walks))
            cov[np.triu_indices(6)] = y[12:]
            cov += np.transpose(np.tril(np.transpose(cov), -1), axes=[1, 2, 0])
            self.walks[i+1] = self.walks[i] + np.array([
                np.random.multivariate_normal(mean=dx[:,j], cov=cov[:,:,j])
                for j in range(n_walks)]).transpose()

        return None


#-----------------------------------------------------------------------------#
