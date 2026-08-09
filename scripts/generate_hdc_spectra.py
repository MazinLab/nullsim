#!/usr/bin/env python
"""Regenerate the packaged HDC template spectra under ``nullsim/data/spectra``.

One-time snapshot generator. The nine CSVs it writes are committed to the
repository and read by `nullsim.physics.spectra.packaged_spectrum_path`; the
test suite never runs this script and never touches the network.

Two resolving powers, kept distinct everywhere
----------------------------------------------

A stored spectrum has a *content* resolving power -- the width of the
line-spread function its features actually carry -- and a *sampling* resolving
power -- how densely that content is written down. Conflating them is how the
first cut of these snapshots shipped R = 30000 content on an R = 30000 grid,
one sample per FWHM, whose recoverable resolving power measures 17-18k. Every
file now stores

``lsf_resolving_power = 30000``
    the Gaussian LSF the samples carry,
``sampling_resolving_power = 90000``
    the constant-R log grid they sit on: three samples per FWHM.

`nullsim.stages.characterization.hdc_ccf` reads ``lsf_resolving_power`` for
every content decision (whether a template out-resolves the analysis grid, how
far the R scan may run) and infers the sampling from the grid itself.

Families and their upstream sources
-----------------------------------

``earth``, ``venus``
    NASA Planetary Spectrum Generator (Villanueva et al. 2018, 2022), POSTed to
    ``https://psg.gsfc.nasa.gov/api.php``. Generation reads a *frozen*,
    committed config under ``scripts/psg_config/`` -- never the service's live
    defaults -- and the SHA-256 of that config goes into the CSV header. The
    frozen config pins the whole ``GENERATOR`` block as literal text, so the
    only per-request overrides are the chunk wavelength bounds (and, for the
    thermal pass, the star). ``--refresh-psg-config`` rebuilds a config from
    scratch; ``--pin-psg-generator`` only expands PSG's generator defaults into
    an existing one, leaving the harvested atmosphere byte-identical.

    The frozen config puts the planet in ``<OBJECT>Exoplanet`` mode at 10 pc
    with the sub-solar point 90 deg from the sub-observer point, which is what
    makes PSG treat the disk as unresolved and run its illuminated-disk
    sub-sampling at quadrature; the named solar-system atmospheres (MERRA-2 for
    Earth, VIRA-45 with volcanic aerosol for Venus) are harvested from
    ``<OBJECT>Planet`` mode and transplanted in, because exoplanet mode
    otherwise falls back to an Earth template for every object.

    Output is ``GENERATOR-RADUNITS=rif``, which PSG labels "I/F [apparent
    albedo]", read from the column named after the object -- not ``Total``,
    which also carries the star. PUMAS refuses 0.5-1.8 um in one shot on memory
    grounds, so the band is requested in overlapping log-spaced chunks; the
    overlaps are compared and the stitch is refused if they disagree.

    PSG's ``rif`` column is *apparent* I/F: reflected sunlight plus the
    planet's own thermal emission, divided by the stellar irradiance. Only the
    reflected part is an albedo -- the thermal part does not scale with the
    host star and must not be carried into a different system. Venus radiates
    through its 1.7 um window at the per cent level, so the Venus template is
    made reflected-only by an explicit subtraction: two extra passes in
    physical radiance units (one full, one with the star extinguished) give the
    thermal fraction sample by sample, which is divided out of the I/F. Earth
    is audited rather than subtracted -- a coarse two-pass check over the red
    end measures its thermal fraction and refuses to write if it is not
    negligible.

    Both are single homogeneous columns integrated over the illuminated disk.
    The Earth file is a clear-column Lambert disk with no clouds, so it is a
    clear-sky endmember rather than a representative Earth; see
    `PSG_TARGET_CAVEAT`, which goes into every header.

``stellar``
    PHOENIX-ACES-AGSS-COND-2011 HiRes FITS (Husser et al. 2013) from
    Goettingen. Downloaded once into ``--cache-dir``, converted from
    erg/s/cm^2/cm to a photon-flux *shape* (multiply by lambda; the absolute
    scale is arbitrary because the consuming stage renormalizes), resampled
    onto a constant-R 300000 grid, degraded from the native R = 500000 to
    R = 30000, then binned onto the output grid. The SHA-256 of every FITS file
    that went in is recorded in the CSV header.

``telluric``
    ESO SkyCalc (Noll et al. 2012; Jones et al. 2013) via the ``skycalc_cli``
    REST client, ``trans`` column: *total* line-of-sight transmission --
    molecular absorption, ozone, Rayleigh and Mie scattering multiplied
    together, not molecular absorption alone. SkyCalc's ``3060m`` site option
    is its Cerro Armazones model, so the Armazones snapshot is native and only
    the Maunakea one is a proxy (4200 m site requested against a 3060 m model);
    the two differ in precipitable water vapour as well.

``radiance``
    The same SkyCalc calls at the same sites, airmass and PWV, keeping the
    ``flux`` column: sky radiance in ph/s/m2/um/arcsec2 from airglow and
    atmospheric thermal emission. Moonlight, zodiacal light and scattered
    starlight are excluded -- nullsim models those separately. Packaged and
    loadable; no stage consumes them yet.

All families are reduced to the same output grid: a constant-resolving-power
log grid at R = 90000 spanning 0.5-1.8 um, produced by
`nullsim.physics.spectra.constant_r_grid`.

Usage::

    conda run -n py313 python scripts/generate_hdc_spectra.py all
    conda run -n py313 python scripts/generate_hdc_spectra.py stellar --only M

``skycalc_cli`` is a script-only dependency (``pip install skycalc_cli``); it
is deliberately absent from ``pyproject.toml`` because the package itself never
regenerates these files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nullsim.physics.spectra import (  # noqa: E402
    constant_r_grid,
    convolve_to_resolving_power,
    resample_flux_conserving,
)

# --------------------------------------------------------------------------
# Output grid
# --------------------------------------------------------------------------

LAMBDA_MIN_UM = 0.5
LAMBDA_MAX_UM = 1.8

# Content resolving power: the Gaussian LSF every family is degraded to.
R_LSF = 30_000.0
# Samples per LSF FWHM on the stored grid. One sample per FWHM -- the first
# cut of these files -- destroys the content: a line stored that way measures
# back out at R ~ 18000. Three is the usual floor for reconstructing a Gaussian
# from point samples and costs only file size.
SAMPLES_PER_FWHM = 3.0
# Sampling resolving power of the stored grid.
R_SAMPLE = R_LSF * SAMPLES_PER_FWHM

OUT_DIR = REPO_ROOT / "nullsim" / "data" / "spectra"

# Every family must supply source samples at least out to the outermost output
# bin edge, so requests are made with this fractional margin on both ends.
REQUEST_MARGIN = 0.01

# --------------------------------------------------------------------------
# PSG (Earth / Venus disk-integrated albedo at quadrature)
# --------------------------------------------------------------------------

PSG_URL = "https://psg.gsfc.nasa.gov/api.php"
PSG_TIMEOUT_S = 1800

# Run PSG targets one at a time. Two of these scripts posting concurrently
# from one machine had a request for Earth answered with a spectrum whose only
# object column was Venus -- the service appears to key working state by
# client rather than by request. `_psg_parse_spectrum` catches the swap, but
# only because it looks the object up by name; do not defeat that by
# parallelising.
PSG_BUSY_MARKER = "Your other API call is still running"
PSG_BUSY_BACKOFF_S = 90.0
# A job abandoned by an interrupted run keeps the slot until the *server*
# finishes it, which for a wide chunk is tens of minutes. The retry budget has
# to outlast that or a regeneration started too soon after a cancelled one
# simply dies.
PSG_BUSY_ATTEMPTS = 60

# The spectra are generated from a *frozen* config committed under
# PSG_CONFIG_DIR, never from PSG's live defaults. `--refresh-psg-config` is the
# only code path that consults the live service for anything but radiance, and
# it rewrites those committed files; running it is a deliberate, reviewable act.
PSG_CONFIG_DIR = Path(__file__).resolve().parent / "psg_config"

# Fixed epoch, used only while refreshing: PSG pulls the Earth atmosphere from
# the NASA/MERRA-2 archive for this date. After the refresh the profile is
# literal text in the frozen config, so the date no longer affects anything.
PSG_DATE = "2026/03/20 12:00"

# Stage 1 of a refresh. PSG only looks up a named solar-system atmosphere in
# `<OBJECT>Planet` mode, so the atmosphere and surface blocks are harvested here
# and transplanted into the exoplanet frame below. Earth resolves to the
# NASA/MERRA-2 profile, Venus to VIRA-45 with the volcanic (sulfuric acid)
# aerosol.
PSG_ATMOSPHERE_SOURCE = {
    name: (
        "<OBJECT>Planet\n"
        f"<OBJECT-NAME>{name.capitalize()}\n"
        f"<OBJECT-DATE>{PSG_DATE}\n"
        "<GEOMETRY>Observatory\n"
    )
    for name in ("earth", "venus")
}

# Stage 2 of a refresh: the observing frame. `<OBJECT>Exoplanet` viewed from
# 10 pc is what makes PSG treat the disk as unresolved and run its illuminated
# -disk sub-sampling (Villanueva et al. 2018, sect. 2.3); in `Planet` mode the
# geometry comes from the solar-system ephemeris and only two disk elements are
# produced. Quadrature is set by placing the sub-solar point 90 deg from the
# sub-observer point -- PSG then returns per-element incidence/emission angles
# and illuminated-area weights that sum to one.
PSG_DISK_ANGLES = 3

PSG_FRAME = {
    # key: (diameter [km], surface gravity [g], semi-major axis [AU])
    "earth": (12742, 9.807, 1.0),
    "venus": (12104, 8.87, 0.723),
}


def psg_frame_text(target: str) -> str:
    diameter, gravity, star_distance_au = PSG_FRAME[target]
    return (
        "<OBJECT>Exoplanet\n"
        f"<OBJECT-NAME>{target.capitalize()}\n"
        f"<OBJECT-DATE>{PSG_DATE}\n"
        f"<OBJECT-DIAMETER>{diameter}\n"
        f"<OBJECT-GRAVITY>{gravity}\n"
        "<OBJECT-GRAVITY-UNIT>g\n"
        "<OBJECT-STAR-TYPE>G\n"
        "<OBJECT-STAR-TEMPERATURE>5777\n"
        "<OBJECT-STAR-RADIUS>1.0\n"
        f"<OBJECT-STAR-DISTANCE>{star_distance_au}\n"
        "<OBJECT-SOLAR-LONGITUDE>90.0\n"
        "<OBJECT-SOLAR-LATITUDE>0.0\n"
        "<OBJECT-OBS-LONGITUDE>0.0\n"
        "<OBJECT-OBS-LATITUDE>0.0\n"
        "<OBJECT-SEASON>90\n"
        "<GEOMETRY>Observatory\n"
        "<GEOMETRY-ALTITUDE-UNIT>pc\n"
        "<GEOMETRY-OBS-ALTITUDE>10.0\n"
        f"<GEOMETRY-DISK-ANGLES>{PSG_DISK_ANGLES}\n"
    )


# The complete GENERATOR block. PSG does not expand its own defaults when it
# echoes a config back -- what you leave out, the live service fills in at
# request time -- so every key the documentation lists is written explicitly,
# including the ones whose value happens to match today's default. That is not
# belt and braces: GENERATOR-CONT-STELLAR flipping from N to Y would stamp the
# host star's absorption lines into the "albedo", changing samples by up to
# 82% and multiplying the stellar spectrum into the planet template twice over,
# and nothing downstream would notice.
PSG_GENERATOR_OVERRIDES = {
    "GENERATOR-RANGEUNIT": "um",
    "GENERATOR-RESOLUTION": f"{R_LSF:.0f}",
    "GENERATOR-RESOLUTIONUNIT": "RP",
    "GENERATOR-RESOLUTIONKERNEL": "Y",
    "GENERATOR-RADUNITS": "rif",
    "GENERATOR-LOGRAD": "N",
    "GENERATOR-GAS-MODEL": "Y",
    "GENERATOR-CONT-MODEL": "Y",
    # No stellar absorption lines in the reflected component: this file is an
    # albedo, and the consuming stage supplies the host spectrum itself.
    "GENERATOR-CONT-STELLAR": "N",
    # No telluric anything -- these are above-the-atmosphere spectra.
    "GENERATOR-TRANS": "N",
    "GENERATOR-TRANS-SHOW": "N",
    "GENERATOR-TRANS-APPLY": "N",
    "GENERATOR-TELESCOPE": "SINGLE",
    "GENERATOR-TELESCOPE1": "1.0",
    "GENERATOR-TELESCOPE2": "1.0",
    "GENERATOR-TELESCOPE3": "1.0",
    "GENERATOR-DIAMTELE": "2.0",
    "GENERATOR-BEAM": "1.0",
    "GENERATOR-BEAM-UNIT": "diffrac",
    "GENERATOR-INSTRUMENT": "user",
    # Noise is off, so the rest of the noise block cannot reach the radiance;
    # pinned anyway so the config is a complete statement of the request.
    "GENERATOR-NOISE": "NONE",
    "GENERATOR-NOISETIME": "1.0",
    "GENERATOR-NOISEFRAMES": "1",
    "GENERATOR-NOISEPIXELS": "8",
    "GENERATOR-NOISE1": "0.0",
    "GENERATOR-NOISE2": "0.0",
    "GENERATOR-NOISEOEFF": "1.0",
    "GENERATOR-NOISEOEMIS": "0.0",
    "GENERATOR-NOISEOTEMP": "273.0",
    "GENERATOR-NOISEWELL": "0",
    "GENERATOR-GCM-BINNING": "1",
}

# With RESOLUTIONKERNEL=Y, PSG convolves to GENERATOR-RESOLUTION and then hands
# the result back sampled ten times finer, i.e. R_sample = 10 * R_LSF. That is
# more than the R_SAMPLE output grid needs, so the rebin below is a pure
# down-bin; the check is asserted at generation time rather than assumed.
PSG_SAMPLING_FACTOR = 10.0

# Physical radiance units for the thermal-separation passes. The I/F pass and
# the two radiance passes share every other setting, so the thermal *fraction*
# they measure is unit-free and can be divided straight out of the I/F.
PSG_RADIANCE_UNITS = "Wsrm2um"

# Thermal-only pass: keep the geometry, the atmosphere and the frozen
# illuminated-disk weights exactly as they are and extinguish the star. What
# comes back is the planet's own emission through the same atmosphere. Moving
# the planet closer to its star instead would be cheaper but is not equivalent
# -- PSG's disk sub-sampling and equilibrium structure both react to it.
PSG_THERMAL_ONLY_OVERRIDES = {
    "OBJECT-STAR-TEMPERATURE": "100",
    "OBJECT-STAR-RADIUS": "1e-6",
}

# Per target: "subtract" runs the two extra full-resolution radiance passes and
# writes reflected-only I/F; "audit" runs a coarse two-pass check over the red
# end and refuses to write if the thermal fraction is not negligible.
PSG_THERMAL_TREATMENT = {"earth": "audit", "venus": "subtract"}

# What each target is and, more usefully, what it is not. PSG's exoplanet mode
# integrates one atmospheric column over the illuminated disk, so whatever
# horizontal structure the real planet has -- clouds above all -- is absent by
# construction. Calling the Earth file a representative Earth would be a
# straightforward overclaim: a clear-column Earth has deeper, cleaner molecular
# bands and a bluer continuum than the real one, whose disk is roughly two
# thirds cloud covered at any moment. It is a defensible clear-sky endmember
# and an indefensible disk average.
PSG_TARGET_CAVEAT = {
    "earth": (
        "single homogeneous clear-column Lambert disk, NOT a representative "
        "Earth. One NASA/MERRA-2 atmospheric column is integrated over the "
        "illuminated disk with a Lambert surface, so there are no clouds, no "
        "land/ocean contrast and no latitudinal structure. Molecular band "
        "depths are therefore clear-sky upper limits and the continuum is a "
        "clear-sky endmember; a cloud-weighted or GCM disk ensemble is future "
        "work"
    ),
    "venus": (
        "single homogeneous column over the illuminated disk with the VIRA-45 "
        "profile and its volcanic (sulfuric acid) aerosol, so there is no "
        "horizontal cloud structure. PSG warns that this configuration's "
        "aerosol phase function would want higher scattering orders "
        "(ATMOSPHERE-NMAX/LMAX) than the frozen config's, which is a known "
        "accuracy limit on the continuum level rather than on the line "
        "positions the cross-correlation uses"
    ),
}

# The audit runs at this resolving power over this band. Thermal fraction is a
# broadband quantity -- a Planck curve divided by a reflected continuum -- so a
# coarse pass measures it for ~1/30 of the cost of a full one.
PSG_AUDIT_RESOLUTION = "1000"
PSG_AUDIT_RANGE_UM = (1.40, 1.82)
# An audited family is only allowed to ship its apparent I/F as an albedo if
# thermal never reaches this fraction of the total anywhere in the audit band.
PSG_AUDIT_MAX_THERMAL_FRACTION = 1.0e-3

# PUMAS memory limit: 0.5-1.8 um in one request is refused. Empirically Earth
# accepts a ~1.4x wavelength span per call and Venus, whose CO2 line list is
# far denser, only ~1.25x. Log-spaced chunks with a 1% overlap on each seam.
PSG_N_CHUNKS = {"earth": 4, "venus": 7}
PSG_CHUNK_OVERLAP = 0.01

# Chunks are predicted twice in their overlap; disagreement there is refused
# rather than quietly written into the template as a seam artefact.
#
# A median-only test is blind to exactly the failure that matters most. Line
# lists are sparse: a chunk boundary that drops or shifts a handful of lines
# moves a few per mille of the samples by order unity and leaves the median
# untouched. So four statistics are computed on every seam, and each has its
# own tolerance:
#
# ``median``   bulk continuum agreement, the original test.
# ``q99``      the 99th percentile -- one sample in a hundred may disagree by
#              this much, which is enough headroom for kernel edge effects.
# ``max``      the single worst sample. Sparse line corruption shows up here
#              first and nowhere else.
# ``line_rms`` the same comparison after both sides are high-pass filtered by
#              dividing out a running median, i.e. on the *lines* alone with
#              the continuum projected out. Normalised by the RMS line
#              contrast, so it answers "what fraction of the line structure
#              disagrees" rather than "what fraction of the flux".
#
# The numbers are calibrated against what PSG actually delivers, not picked
# for roundness. Across all nine seams of the two targets, with the edge guard
# below applied: median <= 8.4e-4, q99 <= 1.7e-2, max <= 2.5e-2,
# line_rms <= 5.8e-3. Chunk edges genuinely disagree at the per cent level in
# a line forest, so a tolerance tight enough to call that an error would only
# cry wolf; these sit a factor of three to five above the real behaviour. What
# they have to catch is a *dropped or shifted* line, which puts a relative
# disagreement of order unity on a handful of samples -- an order of magnitude
# clear of max and line_rms, and invisible to the median that used to be the
# only test.
PSG_SEAM_TOLERANCES = {
    "median": 2.0e-3,
    "q99": 5.0e-2,
    "max": 1.0e-1,
    "line_rms": 3.0e-2,
}
# Resolution elements trimmed from each end of the overlap before comparing.
# PSG truncates its convolution kernel at the request boundary, so the last
# samples of any chunk are attenuated -- at the far edge of an Earth overlap
# the two predictions differ by 44%, entirely from that. Those samples are
# trimmed out of the stitch anyway (the chunk keeps only its own interval,
# hundreds of elements inside the request), so letting them into the seam
# statistic measures nothing except the kernel edge, and would force
# tolerances loose enough to hide anything real.
PSG_SEAM_EDGE_GUARD_ELEMENTS = 50.0
# Relative disagreement is normalised by the larger of the two values, floored
# at this fraction of the overlap's typical level. Without the floor the
# statistic is dominated by saturated line cores, where I/F is three orders
# below the continuum and a numerically irrelevant difference reads as tens of
# per cent -- which would force tolerances so loose that a dropped line, the
# thing worth catching, would slip under them. With it, "10% disagreement"
# means ten per cent of something that matters.
PSG_SEAM_SCALE_FLOOR_FRACTION = 0.05
# Running-median width for the high-pass projection, in samples of the PSG
# grid. One LSF FWHM is PSG_SAMPLING_FACTOR samples there, so 301 is about 30
# resolution elements: wide enough to pass the continuum untouched, narrow
# enough that every individual line lands in the residual.
PSG_SEAM_FILTER_SAMPLES = 301

# --------------------------------------------------------------------------
# PHOENIX (stellar photon-flux shape)
# --------------------------------------------------------------------------

PHOENIX_BASE = "http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS"
PHOENIX_WAVE_FILE = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
PHOENIX_SUFFIX = ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
PHOENIX_SUBDIR = "PHOENIX-ACES-AGSS-COND-2011/Z-0.0"

# Native spectral resolving power of the HiRes grid (Husser et al. 2013).
PHOENIX_R_NATIVE = 500_000.0
# Intermediate constant-R grid: comfortably finer than the R = 30000 kernel, so
# the flux-conserving rebin onto it costs nothing measurable.
PHOENIX_R_WORK = 300_000.0

STELLAR_MODELS = {
    # key: (PHOENIX stem, spectral type, Teff [K], log g, [Fe/H])
    "G": ("lte05800-4.50-0.0", "G2V", 5800, 4.50, 0.0),
    "K": ("lte04900-4.50-0.0", "K2V", 4900, 4.50, 0.0),
    "M": ("lte03500-5.00-0.0", "M2V", 3500, 5.00, 0.0),
}

# --------------------------------------------------------------------------
# SkyCalc (telluric transmission)
# --------------------------------------------------------------------------

# SkyCalc's grid mode is "one sample per lambda/dlambda", and lsf_gauss_fwhm is
# quoted in samples, so requesting R_sample = 300000 with a 10-sample FWHM
# Gaussian LSF delivers a spectrum already degraded to R = 30000.
SKYCALC_R_SAMPLE = 300_000
SKYCALC_LSF_FWHM_BINS = 10.0
# SkyCalc's site options are the ESO sites by altitude; "3060m" is its Cerro
# Armazones model. Armazones is therefore native here and Maunakea is a proxy
# -- a 4200 m site served by the highest model SkyCalc offers.
SKYCALC_OBSERVATORY = "3060m"
SKYCALC_AIRMASS = 1.15

TELLURIC_SITES = {
    # key: (label, PWV [mm], real site altitude [m], native site model?)
    "maunakea": ("Maunakea", 2.5, 4200, False),
    "armazones": ("Cerro Armazones", 5.0, 3046, True),
}

# Emission components kept in the radiance product. Airglow (lines and
# continuum) and atmospheric thermal emission belong to the site; moonlight
# depends on an ephemeris nullsim does not carry, and zodiacal light and
# scattered starlight are modelled elsewhere in the package, so all three stay
# off rather than being double counted.
SKYCALC_RADIANCE_EMISSION = {
    "incl_moon": "N",
    "incl_starlight": "N",
    "incl_zodiacal": "N",
    "incl_loweratm": "Y",
    "incl_upperatm": "Y",
    "incl_airglow": "Y",
    "incl_therm": "Y",
}
SKYCALC_TRANSMISSION_EMISSION = {
    "incl_moon": "N",
    "incl_starlight": "N",
    "incl_zodiacal": "N",
    "incl_loweratm": "Y",
    "incl_upperatm": "Y",
    "incl_airglow": "N",
    "incl_therm": "N",
}

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def output_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the R = 90000 output wavelengths and their bin edges."""
    grid = constant_r_grid(LAMBDA_MIN_UM, LAMBDA_MAX_UM, R_SAMPLE)
    half = np.exp(0.5 / R_SAMPLE)
    edges = np.concatenate([grid / half, [grid[-1] * half]])
    return grid, edges


def check_source_sampling(wavelength_um: np.ndarray, label: str) -> float:
    """Refuse to down-bin a source that is coarser than the output grid.

    Flux-conserving rebinning onto a finer grid than the source carries is
    interpolation wearing a conservation law's clothes: it fills the new
    samples without adding content, and the file then advertises a sampling it
    does not have. Every family has to arrive at least as densely sampled as
    `R_SAMPLE`.
    """
    dln = np.diff(np.log(np.asarray(wavelength_um, dtype=np.float64)))
    r_source = 1.0 / float(np.median(dln))
    if r_source < R_SAMPLE:
        raise RuntimeError(
            f"{label}: source is sampled at R={r_source:.4g}, below the "
            f"R={R_SAMPLE:.0f} output grid; rebinning would invent samples"
        )
    return r_source


def write_csv(
    path: Path,
    wavelength_um: np.ndarray,
    values: np.ndarray,
    kind: str,
    provenance: dict[str, str],
) -> None:
    """Write a provenance-headed ``wavelength_um,value`` CSV."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name}: non-finite values")
    if kind == "transmission" and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(f"{path.name}: transmission outside [0, 1]")
    if values.min() < 0.0:
        raise ValueError(f"{path.name}: negative values")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {key}: {value}" for key, value in provenance.items()]
    lines.append("wavelength_um,value")
    lines.extend(
        f"{w:.10e},{v:.6e}" for w, v in zip(wavelength_um, values, strict=True)
    )
    path.write_text("\n".join(lines) + "\n")
    print(
        f"  wrote {path.relative_to(REPO_ROOT)}  "
        f"{values.size} rows, {path.stat().st_size / 1e6:.2f} MB, "
        f"value range [{values.min():.4g}, {values.max():.4g}]"
    )


def base_provenance(kind: str, source: str, command: str) -> dict[str, str]:
    return {
        "kind": kind,
        "source": source,
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        # Content resolution and grid density are separate facts about a
        # snapshot and are recorded separately. Consumers deciding what the
        # spectrum resolves want lsf_resolving_power; the sampling is also
        # recoverable from the grid itself, and the two are cross-checked by
        # the snapshot tests.
        "lsf_resolving_power": f"{R_LSF:.0f}",
        "sampling_resolving_power": f"{R_SAMPLE:.0f}",
        "samples_per_fwhm": f"{SAMPLES_PER_FWHM:g}",
        "wavelength_range_um": f"{LAMBDA_MIN_UM}-{LAMBDA_MAX_UM}",
        "grid": "constant-R log grid (constant_r_grid)",
        "generator_command": command,
    }


# --------------------------------------------------------------------------
# PSG albedo
# --------------------------------------------------------------------------


_PSG_CACHE_DIR: Path | None = None


def set_psg_cache_dir(path: Path | None) -> None:
    """Point PSG response caching at a scratch directory (never the repo).

    A full band costs tens of minutes of radiative transfer, and the service
    allows one job at a time, so a reduction bug found after the last chunk
    lands used to mean fetching the whole band again. Responses are keyed by
    the exact request body, so a cached hit is the same bytes the service
    would return -- and any change to the frozen config or the chunking
    invalidates the key by construction.
    """
    global _PSG_CACHE_DIR
    _PSG_CACHE_DIR = path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def _psg_cache_path(payload: dict[str, str]) -> Path | None:
    if _PSG_CACHE_DIR is None:
        return None
    key = hashlib.sha256(
        "\x00".join(f"{k}={payload[k]}" for k in sorted(payload)).encode()
    ).hexdigest()
    return _PSG_CACHE_DIR / f"psg_{key}.txt"


def _psg_post(payload: dict[str, str], attempts: int = PSG_BUSY_ATTEMPTS) -> str:
    """POST to PSG, waiting out the one-job-per-client limit.

    PSG answers a second concurrent request from the same client with a plain
    sentence rather than an ERROR line -- and if an earlier run was killed, its
    job keeps occupying the slot for a while afterwards. Without this the
    sentence flows into the spectrum parser and surfaces as a bewildering
    "data rows before a column header".
    """
    cache_path = _psg_cache_path(payload)
    if cache_path is not None and cache_path.exists():
        print(f"    cached PSG response {cache_path.name}", flush=True)
        return cache_path.read_text()

    for attempt in range(1, attempts + 1):
        response = requests.post(PSG_URL, data=payload, timeout=PSG_TIMEOUT_S)
        response.raise_for_status()
        text = response.text
        if PSG_BUSY_MARKER in text:
            if attempt == attempts:
                raise RuntimeError(
                    f"PSG still busy after {attempts} attempts: {text.strip()[:200]}"
                )
            print(
                f"    PSG busy (attempt {attempt}/{attempts}); "
                f"retrying in {PSG_BUSY_BACKOFF_S:.0f} s",
                flush=True,
            )
            time.sleep(PSG_BUSY_BACKOFF_S)
            continue
        errors = [ln for ln in text.splitlines() if "ERROR" in ln.upper()]
        if errors:
            raise RuntimeError("PSG reported an error: " + "; ".join(errors[:3]))
        if cache_path is not None:
            cache_path.write_text(text)
        return text
    raise RuntimeError("unreachable")


def psg_config_path(target: str) -> Path:
    return PSG_CONFIG_DIR / f"{target}_quadrature.cfg"


def _psg_apply_overrides(config: str, overrides: dict[str, str]) -> str:
    kept = [
        line
        for line in config.splitlines()
        if not any(line.startswith(f"<{key}>") for key in overrides)
    ]
    kept.extend(f"<{key}>{value}" for key, value in overrides.items())
    return "\n".join(kept) + "\n"


def refresh_psg_config(target: str) -> Path:
    """Rebuild and freeze the fully expanded PSG config for ``target``.

    The only code path that reads PSG's live defaults. Stage 1 harvests the
    named atmosphere and surface blocks from ``<OBJECT>Planet`` mode; stage 2
    transplants them into the exoplanet quadrature frame and lets PSG's geometry
    module resolve the illuminated-disk sub-sampling. The result is written to
    ``scripts/psg_config/`` and committed, so generation never depends on the
    service's defaults again.
    """
    print(f"[{target}] refreshing frozen PSG config")
    print("  stage 1: harvesting the named atmosphere from Planet mode")
    planet_cfg = _psg_post(
        {
            "type": "cfg",
            "wephm": "n",
            "watm": "y",
            "file": PSG_ATMOSPHERE_SOURCE[target],
        }
    )
    blocks = [
        line
        for line in planet_cfg.splitlines()
        if line.startswith(("<ATMOSPHERE-", "<SURFACE-"))
    ]
    if not blocks:
        raise RuntimeError(f"PSG returned no atmosphere block for {target}")
    description = next(
        (b for b in blocks if b.startswith("<ATMOSPHERE-DESCRIPTION>")), "<unknown>"
    )
    print(f"    {description[:100]}")

    print("  stage 2: expanding the exoplanet quadrature frame")
    frame = _psg_apply_overrides(
        psg_frame_text(target) + "\n".join(blocks) + "\n", PSG_GENERATOR_OVERRIDES
    )
    expanded = _psg_post(
        {"type": "cfg", "wephm": "n", "watm": "n", "wgeo": "y", "file": frame}
    )
    fractions = _psg_disk_fractions(expanded)
    total = float(np.sum(fractions))
    print(
        f"    {fractions.size} illuminated-disk elements, "
        f"area weights sum to {total:.4f}"
    )
    if fractions.size < 2:
        raise RuntimeError(
            f"{target}: PSG returned {fractions.size} disk element(s); the "
            "illuminated-disk integration did not engage"
        )
    if not 0.95 <= total <= 1.05:
        raise RuntimeError(
            f"{target}: disk area weights sum to {total:.4f}, expected ~1"
        )

    path = psg_config_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expanded)
    print(f"  froze {path.relative_to(REPO_ROOT)}  sha256={_sha256(path)[:16]}...")
    return path


def pin_psg_generator(target: str) -> Path:
    """Expand PSG's generator defaults into the existing frozen config.

    ``refresh_psg_config`` rebuilds a config from the live service, atmosphere
    and all; this only asks PSG to write out the ``GENERATOR`` block it would
    have used, so a config frozen before the block was pinned can be brought up
    to date without re-harvesting -- and without the atmosphere moving under
    the spectra it already produced. The harvested atmosphere, surface and
    geometry blocks are literal text in the file and are asserted unchanged.
    """
    path = psg_config_path(target)
    if not path.exists():
        raise SystemExit(f"missing frozen PSG config {path}")
    before = path.read_text()
    print(f"[{target}] pinning the PSG generator block into {path.name}")
    expanded = _psg_post(
        {
            "type": "cfg",
            "wephm": "n",
            "watm": "n",
            "wgeo": "n",
            "file": _psg_apply_overrides(before, PSG_GENERATOR_OVERRIDES),
        }
    )
    kept_prefixes = ("<ATMOSPHERE-", "<SURFACE-", "<GEOMETRY-", "<OBJECT")
    old_blocks = [ln for ln in before.splitlines() if ln.startswith(kept_prefixes)]
    new_blocks = [ln for ln in expanded.splitlines() if ln.startswith(kept_prefixes)]
    missing = [ln for ln in old_blocks if ln not in new_blocks]
    if missing:
        raise RuntimeError(
            f"{target}: PSG altered {len(missing)} frozen line(s) while "
            f"expanding the generator block, first {missing[0][:80]!r}"
        )
    path.write_text(expanded)
    generator_keys = [
        ln for ln in expanded.splitlines() if ln.startswith("<GENERATOR-")
    ]
    print(
        f"  froze {len(generator_keys)} GENERATOR keys, "
        f"sha256={_sha256(path)[:16]}..."
    )
    return path


def _psg_config_generator_settings(config: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for line in config.splitlines():
        if line.startswith("<GENERATOR-"):
            key, _, value = line[1:].partition(">")
            settings[key] = value
    return settings


def _check_frozen_generator(target: str, config: str) -> dict[str, str]:
    """Assert the frozen config already carries the generator settings used.

    The point of freezing is that nothing about the request comes from the
    service's live defaults. If the committed file is missing a key, the
    request would silently pick one up, so refuse instead of overriding.
    """
    settings = _psg_config_generator_settings(config)
    wrong = {
        key: (settings.get(key), value)
        for key, value in PSG_GENERATOR_OVERRIDES.items()
        if settings.get(key) != value
    }
    if wrong:
        raise SystemExit(
            f"{target}: the frozen PSG config does not pin "
            f"{sorted(wrong)} as expected ({wrong}). Run with "
            "--pin-psg-generator to expand PSG's generator defaults into it."
        )
    return settings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _psg_disk_fractions(config: str) -> np.ndarray:
    for line in config.splitlines():
        if line.startswith("<GEOMETRY-PLANET-FRACTION>"):
            body = line.split(">", 1)[1]
            return np.asarray([float(x) for x in body.split(",")], dtype=np.float64)
    return np.zeros(0, dtype=np.float64)


def _psg_parse_spectrum(text: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull the wavelength and *planet* columns out of a PSG rad file.

    In exoplanet mode PSG emits ``Wave Total Noise Stellar <ObjectName>``. The
    ``Total`` column is the star plus the planet, and since the star outshines
    the planet by ~1e10 in the beam it is numerically meaningless here (it comes
    back oscillating over +-1e4 in I/F units). The planet's own contribution is
    the column named after the object, which is what an albedo means.
    """
    columns: list[str] | None = None
    waves: list[float] = []
    values: list[float] = []
    index: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if body.startswith("Wave"):
                columns = body.split()
                wanted = target.capitalize()
                if wanted in columns:
                    index = columns.index(wanted)
                elif "Object" in columns:
                    index = columns.index("Object")
                else:
                    raise RuntimeError(
                        f"no planet column for {target!r} in PSG columns {columns}"
                    )
            continue
        if not stripped:
            continue
        if index is None:
            raise RuntimeError("PSG returned data rows before a column header")
        fields = stripped.split()
        waves.append(float(fields[0]))
        values.append(float(fields[index]))
    if not waves:
        raise RuntimeError("PSG returned no spectral rows")
    return np.asarray(waves, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _psg_chunk_edges(n_chunks: int) -> np.ndarray:
    lo = LAMBDA_MIN_UM * (1.0 - REQUEST_MARGIN)
    hi = LAMBDA_MAX_UM * (1.0 + REQUEST_MARGIN)
    return lo * (hi / lo) ** (np.arange(n_chunks + 1) / n_chunks)


def _psg_request_chunks(
    target: str,
    config: str,
    bounds: np.ndarray,
    extra_overrides: dict[str, str],
    label: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """POST one PSG request per chunk and return the raw planet spectra."""
    n_chunks = bounds.size - 1
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_chunks):
        req_lo = bounds[i] * (1.0 - PSG_CHUNK_OVERLAP)
        req_hi = bounds[i + 1] * (1.0 + PSG_CHUNK_OVERLAP)
        print(
            f"  PSG {label}: chunk {i + 1}/{n_chunks} "
            f"{req_lo:.4f}-{req_hi:.4f} um",
            flush=True,
        )
        overrides = dict(extra_overrides)
        overrides["GENERATOR-RANGE1"] = f"{req_lo:.6f}"
        overrides["GENERATOR-RANGE2"] = f"{req_hi:.6f}"
        text = _psg_post(
            {
                "type": "rad",
                "wephm": "n",
                "wgeo": "n",
                "file": _psg_apply_overrides(config, overrides),
            }
        )
        raw.append(_psg_parse_spectrum(text, target))
    return raw


def _high_pass(values: np.ndarray) -> np.ndarray:
    """Divide out a running median: the continuum-projected line residual."""
    from scipy.ndimage import median_filter

    width = min(PSG_SEAM_FILTER_SAMPLES, max(3, (values.size // 2) * 2 - 1))
    continuum = median_filter(values, size=width, mode="nearest")
    safe = np.where(np.abs(continuum) > 0.0, continuum, 1.0)
    return values / safe - 1.0


def _seam_metrics(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[dict[str, float], int]:
    """Compare two chunks where they overlap, on the continuum and the lines.

    See `PSG_SEAM_TOLERANCES` for why four numbers rather than one.
    """
    left_w, left_v = left
    right_w, right_v = right
    guard = math.exp(PSG_SEAM_EDGE_GUARD_ELEMENTS / R_LSF)
    lo = max(left_w[0], right_w[0]) * guard
    hi = min(left_w[-1], right_w[-1]) / guard
    if hi <= lo:
        raise RuntimeError("chunks do not overlap outside their kernel edges")
    probe = left_w[(left_w >= lo) & (left_w <= hi)]
    if probe.size < 100:
        raise RuntimeError(
            f"only {probe.size} samples survive the seam edge guard; the "
            "chunk overlap is too narrow to validate"
        )
    mine = np.interp(probe, left_w, left_v)
    other = np.interp(probe, right_w, right_v)
    floor = PSG_SEAM_SCALE_FLOOR_FRACTION * float(np.median(np.abs(mine)))
    scale = np.maximum(np.maximum(np.abs(mine), np.abs(other)), floor)
    good = scale > 0.0
    rel = np.abs(mine - other)[good] / scale[good]

    mine_lines = _high_pass(mine)
    other_lines = _high_pass(other)
    line_contrast = float(np.sqrt(np.mean(mine_lines**2)))
    line_rms = float(np.sqrt(np.mean((mine_lines - other_lines) ** 2)))
    metrics = {
        "median": float(np.median(rel)) if rel.size else 0.0,
        "q99": float(np.quantile(rel, 0.99)) if rel.size else 0.0,
        "max": float(np.max(rel)) if rel.size else 0.0,
        "line_rms": line_rms / line_contrast if line_contrast > 0.0 else 0.0,
    }
    return metrics, probe.size


def _validate_seams(
    target: str, raw: list[tuple[np.ndarray, np.ndarray]], bounds: np.ndarray
) -> dict[str, float]:
    worst = dict.fromkeys(PSG_SEAM_TOLERANCES, 0.0)
    for i in range(len(raw) - 1):
        metrics, n_probe = _seam_metrics(raw[i], raw[i + 1])
        print(
            f"    seam {i + 1} at {bounds[i + 1]:.4f} um: {n_probe} overlapping "
            + ", ".join(f"{k} {v:.3e}" for k, v in metrics.items())
        )
        for key, value in metrics.items():
            worst[key] = max(worst[key], value)
            if value > PSG_SEAM_TOLERANCES[key]:
                raise RuntimeError(
                    f"{target}: seam {i + 1} at {bounds[i + 1]:.4f} um has "
                    f"{key} disagreement {value:.3e} > "
                    f"{PSG_SEAM_TOLERANCES[key]:.0e}; refusing to stitch"
                )
    return worst


def _stitch(
    raw: list[tuple[np.ndarray, np.ndarray]], bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trim each chunk back to its own interval and concatenate."""
    n_chunks = len(raw)
    pieces = []
    for i, (wave, value) in enumerate(raw):
        keep = np.ones(wave.shape, dtype=bool)
        if i > 0:
            keep &= wave >= bounds[i]
        if i < n_chunks - 1:
            keep &= wave < bounds[i + 1]
        pieces.append((wave[keep], value[keep]))
    wavelength = np.concatenate([p[0] for p in pieces])
    values = np.concatenate([p[1] for p in pieces])
    order = np.argsort(wavelength)
    wavelength, values = wavelength[order], values[order]
    unique = np.concatenate([[True], np.diff(wavelength) > 0.0])
    return wavelength[unique], values[unique]


def _thermal_fraction(
    target: str, config: str, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample-by-sample thermal share of the planet's radiance.

    Two passes in physical radiance units -- one as configured, one with the
    star extinguished -- over the same grid. Radiative transfer is linear in
    its sources, so the second pass is exactly the thermal term of the first,
    and their ratio is the fraction of the apparent I/F that is emission rather
    than reflection. Working in a ratio is what lets the correction be applied
    to the ``rif`` pass without ever needing PSG's I/F normalisation.
    """
    radiance_overrides = {"GENERATOR-RADUNITS": PSG_RADIANCE_UNITS}
    total_w, total_v = _stitch(
        _psg_request_chunks(
            target, config, bounds, radiance_overrides, "radiance total"
        ),
        bounds,
    )
    thermal_w, thermal_v = _stitch(
        _psg_request_chunks(
            target,
            config,
            bounds,
            {**radiance_overrides, **PSG_THERMAL_ONLY_OVERRIDES},
            "radiance thermal",
        ),
        bounds,
    )
    thermal_on_total = np.interp(total_w, thermal_w, thermal_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(total_v > 0.0, thermal_on_total / total_v, 0.0)
    fraction = np.clip(np.nan_to_num(fraction, nan=0.0), 0.0, 1.0)
    return total_w, fraction


def _audit_thermal_fraction(target: str, config: str) -> float:
    """Coarse two-pass measurement of the thermal share over the red end."""
    lo, hi = PSG_AUDIT_RANGE_UM
    bounds = np.asarray([lo, hi], dtype=np.float64)
    coarse = {"GENERATOR-RESOLUTION": PSG_AUDIT_RESOLUTION}
    _, fraction = _thermal_fraction(
        target,
        _psg_apply_overrides(config, coarse),
        bounds,
    )
    worst = float(np.max(fraction))
    print(
        f"  thermal audit {lo}-{hi} um at R={PSG_AUDIT_RESOLUTION}: "
        f"max thermal fraction {worst:.3e}"
    )
    if worst > PSG_AUDIT_MAX_THERMAL_FRACTION:
        raise RuntimeError(
            f"{target}: thermal emission reaches {worst:.3e} of the apparent "
            f"I/F, above the {PSG_AUDIT_MAX_THERMAL_FRACTION:.0e} an albedo "
            "template may carry. Switch this target to the 'subtract' "
            "treatment."
        )
    return worst


def generate_albedo(target: str, command: str) -> None:
    print(f"[{target}] PSG disk-integrated albedo at quadrature")
    path = psg_config_path(target)
    if not path.exists():
        raise SystemExit(
            f"missing frozen PSG config {path}. Run with --refresh-psg-config "
            "to rebuild it from the service (this changes committed inputs)."
        )
    config = path.read_text()
    config_hash = _sha256(path)
    _check_frozen_generator(target, config)
    fractions = _psg_disk_fractions(config)
    print(
        f"  frozen config {path.name} sha256={config_hash[:16]}... "
        f"{fractions.size} disk elements"
    )

    n_chunks = PSG_N_CHUNKS[target]
    bounds = _psg_chunk_edges(n_chunks)
    raw = _psg_request_chunks(target, config, bounds, {}, "I/F")

    # Chunks overlap by PSG_CHUNK_OVERLAP on each side of every seam, so each
    # seam has a region predicted twice. Disagreement there is the direct
    # measure of whether stitching is safe; a silent seam artefact would inject
    # a fake spectral feature straight into the CCF template.
    seam_worst = _validate_seams(target, raw, bounds)
    wavelength, albedo = _stitch(raw, bounds)
    r_source = check_source_sampling(wavelength, f"{target} PSG")
    expected = PSG_SAMPLING_FACTOR * R_LSF
    if not 0.8 * expected <= r_source <= 1.25 * expected:
        raise RuntimeError(
            f"{target}: PSG returned R={r_source:.4g} sampling where "
            f"RESOLUTIONKERNEL=Y should give R={expected:.4g}; the request "
            "did not carry the generator settings the config pins"
        )

    treatment = PSG_THERMAL_TREATMENT[target]
    if treatment == "subtract":
        thermal_w, thermal_fraction = _thermal_fraction(target, config, bounds)
        fraction_on_grid = np.interp(wavelength, thermal_w, thermal_fraction)
        worst_thermal = float(np.max(fraction_on_grid))
        albedo = albedo * (1.0 - fraction_on_grid)
        thermal_note = (
            "reflected only: the thermal share of the planet's radiance was "
            "measured sample by sample with two extra PSG passes in "
            f"{PSG_RADIANCE_UNITS} (one as configured, one with the star "
            "extinguished via "
            + ", ".join(f"{k}={v}" for k, v in PSG_THERMAL_ONLY_OVERRIDES.items())
            + f") and divided out; it peaks at {worst_thermal:.3e} of the "
            "apparent I/F"
        )
        print(f"  reflected-only: peak thermal fraction {worst_thermal:.3e}")
    elif treatment == "audit":
        worst_thermal = _audit_thermal_fraction(target, config)
        thermal_note = (
            "reflected only to within the audit: a coarse "
            f"R={PSG_AUDIT_RESOLUTION} two-pass check over "
            f"{PSG_AUDIT_RANGE_UM[0]}-{PSG_AUDIT_RANGE_UM[1]} um puts the "
            f"thermal share of the apparent I/F at {worst_thermal:.3e} at "
            "worst, below the "
            f"{PSG_AUDIT_MAX_THERMAL_FRACTION:.0e} threshold, so no "
            "subtraction is applied"
        )
    else:
        raise ValueError(f"unknown thermal treatment {treatment!r}")

    grid, edges = output_grid()
    resampled = resample_flux_conserving(wavelength, albedo, edges)

    n_clipped = int(np.count_nonzero((resampled < 0.0) | (resampled > 1.0)))
    if n_clipped > 0.01 * resampled.size:
        raise RuntimeError(
            f"{target}: {n_clipped} of {resampled.size} samples fall outside "
            "[0, 1]; that is too many to write off as band-edge noise"
        )
    resampled = np.clip(resampled, 0.0, 1.0)

    provenance = base_provenance(
        "albedo",
        "NASA Planetary Spectrum Generator (Villanueva et al. 2018, 2022), "
        f"api.php, object {target.capitalize()}",
        command,
    )
    provenance.update(
        {
            "psg_config": f"scripts/psg_config/{path.name}",
            "psg_config_sha256": config_hash,
            "psg_geometry": (
                "Exoplanet at 10 pc, sub-solar point 90 deg from the "
                "sub-observer point => phase angle 90 deg (quadrature); PSG "
                f"illuminated-disk sub-sampling with DISK-ANGLES="
                f"{PSG_DISK_ANGLES} giving {fractions.size} elements, "
                f"area weights summing to {float(np.sum(fractions)):.4f}"
            ),
            "psg_generator": (
                f"RADUNITS=rif (I/F apparent albedo), RESOLUTION={R_LSF:.0f} "
                "RP, RESOLUTIONKERNEL=Y, GAS-MODEL=Y, CONT-MODEL=Y, "
                "NOISE=NONE, all pinned as literal text in the frozen config; "
                f"planet column '{target.capitalize()}' (not 'Total', which "
                f"includes the star); requested in {n_chunks} overlapping "
                "log-spaced chunks (PUMAS refuses the full band in one call) "
                "and stitched"
            ),
            "psg_max_seam_disagreement": (
                ", ".join(
                    f"{key} {value:.3e} (tol {PSG_SEAM_TOLERANCES[key]:.0e})"
                    for key, value in seam_worst.items()
                )
                + "; relative, worst seam, on the raw samples and on the "
                "continuum-filtered line residual"
            ),
            # Machine-readable companions to thermal_emission: what was done
            # about the emission term, and how large it was. Peak thermal
            # fraction is the number that decides whether calling this file an
            # albedo is defensible, so it is recorded as a number rather than
            # only inside a sentence.
            "thermal_treatment": treatment,
            "thermal_fraction_peak": f"{worst_thermal:.6e}",
            "thermal_emission": thermal_note,
            "reduction": (
                f"PSG returns the R={R_LSF:.0f} kernel sampled at "
                f"R={r_source:.0f}; flux-conserving rebin onto the "
                f"R={R_SAMPLE:.0f} output grid"
            ),
            "model_caveat": PSG_TARGET_CAVEAT[target],
            "note": (
                "disk-integrated albedo at quadrature phase, not the "
                f"zero-phase geometric albedo; {n_clipped} samples clipped to "
                "[0, 1]"
            ),
        }
    )
    write_csv(
        OUT_DIR / f"{target}_albedo.csv", grid, resampled, "albedo", provenance
    )


# --------------------------------------------------------------------------
# PHOENIX stellar
# --------------------------------------------------------------------------


def _download(url: str, dest: Path, attempts: int = 5) -> Path:
    """Fetch ``url`` to ``dest``, reusing an existing download.

    The Goettingen archive intermittently answers 502 through its proxy, so
    failed attempts are retried with a short backoff.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        print(f"  downloading {url} (attempt {attempt}/{attempts})")
        try:
            with requests.get(url, stream=True, timeout=1800) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
            return dest
        except requests.RequestException as exc:
            dest.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f"    failed ({exc}); retrying in 10 s")
            time.sleep(10.0)
    return dest


def generate_stellar(keys: list[str], cache_dir: Path, command: str) -> None:
    from astropy.io import fits

    wave_path = _download(
        f"{PHOENIX_BASE}/{PHOENIX_WAVE_FILE}", cache_dir / PHOENIX_WAVE_FILE
    )
    with fits.open(wave_path) as hdul:
        wave_um_all = np.asarray(hdul[0].data, dtype=np.float64) * 1e-4
    wave_sha256 = _sha256(wave_path)

    grid, edges = output_grid()
    lo = LAMBDA_MIN_UM * (1.0 - REQUEST_MARGIN)
    hi = LAMBDA_MAX_UM * (1.0 + REQUEST_MARGIN)
    window = (wave_um_all >= lo) & (wave_um_all <= hi)
    wave_um = wave_um_all[window]

    work_grid = constant_r_grid(lo, hi, PHOENIX_R_WORK)
    work_half = np.exp(0.5 / PHOENIX_R_WORK)
    work_edges = np.concatenate([work_grid / work_half, [work_grid[-1] * work_half]])
    work_edges = work_edges[
        (work_edges >= wave_um[0]) & (work_edges <= wave_um[-1])
    ]
    work_grid = np.sqrt(work_edges[:-1] * work_edges[1:])

    for key in keys:
        stem, sptype, teff, logg, feh = STELLAR_MODELS[key]
        print(f"[stellar {key}] PHOENIX {sptype} {stem}")
        flux_path = _download(
            f"{PHOENIX_BASE}/{PHOENIX_SUBDIR}/{stem}{PHOENIX_SUFFIX}",
            cache_dir / f"{stem}{PHOENIX_SUFFIX}",
        )
        with fits.open(flux_path) as hdul:
            flux_all = np.asarray(hdul[0].data, dtype=np.float64)
        flux_sha256 = _sha256(flux_path)
        # erg/s/cm^2/cm -> photon-flux shape: F_lambda * lambda / (h c). The
        # constant is dropped; the consuming stage renormalizes.
        photon = flux_all[window] * wave_um

        work = resample_flux_conserving(wave_um, photon, work_edges)
        degraded = convolve_to_resolving_power(
            work_grid, work, R_LSF, r_in=PHOENIX_R_NATIVE
        )
        check_source_sampling(work_grid, f"stellar {key} work grid")
        resampled = resample_flux_conserving(work_grid, degraded, edges)
        # PHOENIX fluxes are physically non-negative; anything below zero here
        # is our own numerics, and a *lot* of it would mean the reduction is
        # wrong rather than merely noisy. Never floor silently.
        negative = resampled < 0.0
        n_negative = int(np.count_nonzero(negative))
        worst = float(resampled.min()) if n_negative else 0.0
        if n_negative:
            median_positive = float(np.median(resampled[~negative]))
            if (
                n_negative > 0.001 * resampled.size
                or abs(worst) > 1e-6 * median_positive
            ):
                raise RuntimeError(
                    f"{stem}: {n_negative} of {resampled.size} resampled "
                    f"samples are negative (min {worst:.3e} against a median "
                    f"of {median_positive:.3e}); the reduction is suspect"
                )
            print(
                f"  warning: floored {n_negative} negative sample(s), "
                f"min {worst:.3e}"
            )
            resampled = np.maximum(resampled, 0.0)

        provenance = base_provenance(
            "stellar_flux",
            "PHOENIX-ACES-AGSS-COND-2011 HiRes (Husser et al. 2013, A&A 553, "
            f"A6), {stem}{PHOENIX_SUFFIX}",
            command,
        )
        provenance.update(
            {
                "phoenix_model": (
                    f"{sptype}: Teff={teff} K, log g={logg}, [Fe/H]={feh}, "
                    "alpha=0"
                ),
                "phoenix_archive": f"{PHOENIX_BASE}/{PHOENIX_SUBDIR}",
                # The FITS files are far too large to commit, so the bytes that
                # produced this snapshot are bound by digest instead: anyone
                # re-downloading from Goettingen can confirm they hold the same
                # input before believing they reproduced the output.
                "phoenix_flux_sha256": f"{stem}{PHOENIX_SUFFIX} {flux_sha256}",
                "phoenix_wave_sha256": f"{PHOENIX_WAVE_FILE} {wave_sha256}",
                "units": (
                    "arbitrary photon-flux-density shape (F_lambda * lambda, "
                    "h c dropped); the consuming stage renormalizes"
                ),
                "wavelengths": "vacuum, from WAVE_PHOENIX-ACES-AGSS-COND-2011",
                "reduction": (
                    f"flux-conserving rebin onto a constant-R "
                    f"{PHOENIX_R_WORK:.0f} grid, Gaussian LSF degradation "
                    f"R={PHOENIX_R_NATIVE:.0f} -> R={R_LSF:.0f}, "
                    f"flux-conserving rebin onto the R={R_SAMPLE:.0f} output "
                    "grid"
                ),
                "negative_samples_floored": (
                    f"{n_negative} (most negative {worst:.3e})"
                ),
            }
        )
        write_csv(
            OUT_DIR / f"stellar_{sptype.lower()}.csv",
            grid,
            resampled,
            "stellar_flux",
            provenance,
        )


# --------------------------------------------------------------------------
# SkyCalc transmission and sky radiance
# --------------------------------------------------------------------------


def _skycalc_call(
    key: str, product: str, pwv: float, cache_dir: Path
) -> tuple[object, np.ndarray, dict[str, np.ndarray], str]:
    """Run one SkyCalc request and return the model, grid, columns and unit."""
    from astropy.io import fits
    from skycalc_cli.skycalc import SkyModel

    wmin_nm = LAMBDA_MIN_UM * (1.0 - REQUEST_MARGIN) * 1e3
    wmax_nm = LAMBDA_MAX_UM * (1.0 + REQUEST_MARGIN) * 1e3
    emission = (
        SKYCALC_RADIANCE_EMISSION
        if product == "radiance"
        else SKYCALC_TRANSMISSION_EMISSION
    )
    model = SkyModel()
    model.stop_on_errors_and_exceptions = False
    model.params.update(
        {
            "airmass": SKYCALC_AIRMASS,
            "pwv_mode": "pwv",
            "pwv": pwv,
            "observatory": SKYCALC_OBSERVATORY,
            **emission,
            "vacair": "vac",
            "wmin": wmin_nm,
            "wmax": wmax_nm,
            "wgrid_mode": "fixed_spectral_resolution",
            "wres": SKYCALC_R_SAMPLE,
            "lsf_type": "Gaussian",
            "lsf_gauss_fwhm": SKYCALC_LSF_FWHM_BINS,
        }
    )
    model.call()
    raw = cache_dir / f"skycalc_{product}_{key}.fits"
    raw.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(raw))

    with fits.open(raw) as hdul:
        table = hdul[1].data
        lam_unit = str(
            next(c.unit for c in hdul[1].columns if c.name == "lam")
        ).strip()
        columns = {
            name: np.asarray(table[name], dtype=np.float64)
            for name in ("lam", "trans", "flux")
        }

    # skycalc_cli changed the output wavelength unit between v1.3 (um) and
    # v1.4 (nm), so read it from the table rather than assuming, and then
    # confirm the coverage actually matches what was requested.
    scale = {"nm": 1e-3, "um": 1.0, "micron": 1.0}.get(lam_unit)
    if scale is None:
        raise RuntimeError(f"SkyCalc returned an unrecognised 'lam' unit {lam_unit!r}")
    wave_um = columns["lam"] * scale
    if not (
        abs(wave_um[0] - wmin_nm * 1e-3) < 0.02
        and abs(wave_um[-1] - wmax_nm * 1e-3) < 0.02
    ):
        raise RuntimeError(
            f"SkyCalc coverage [{wave_um[0]:.4f}, {wave_um[-1]:.4f}] um does "
            f"not match the requested [{wmin_nm * 1e-3:.4f}, "
            f"{wmax_nm * 1e-3:.4f}] um (unit read as {lam_unit!r})"
        )
    print(
        f"  SkyCalc returned {wave_um.size} samples, 'lam' unit {lam_unit!r}, "
        f"coverage {wave_um[0]:.4f}-{wave_um[-1]:.4f} um"
    )
    check_source_sampling(wave_um, f"skycalc {product} {key}")
    return model, wave_um, columns, lam_unit


def _skycalc_provenance(
    key: str,
    kind: str,
    model: object,
    wave_um: np.ndarray,
    lam_unit: str,
    command: str,
) -> dict[str, str]:
    from importlib.metadata import version

    label, pwv, altitude_m, native = TELLURIC_SITES[key]
    provenance = base_provenance(
        kind,
        "ESO SkyCalc / Cerro Paranal Advanced Sky Model (Noll et al. 2012, "
        "A&A 543, A92; Jones et al. 2013, A&A 560, A91) via skycalc_cli",
        command,
    )
    if native:
        site = (
            f"{label} ({altitude_m} m). SkyCalc's observatory="
            f"{SKYCALC_OBSERVATORY} option IS its Cerro Armazones site model, "
            "so this is the native model and not a substitute; the "
            "precipitable water vapour is set explicitly rather than taken "
            "from the site default"
        )
    else:
        site = (
            f"proxy for {label} ({altitude_m} m). SkyCalc offers no Maunakea "
            f"model, so the request uses observatory={SKYCALC_OBSERVATORY} -- "
            "its highest site, Cerro Armazones at 3060 m -- with Maunakea's "
            "precipitable water vapour. Above ~3 km the residual altitude "
            "difference mostly rescales the dry-air column, which this does "
            "not correct for"
        )
    provenance.update(
        {
            "site": site,
            "skycalc_client": f"skycalc_cli {version('skycalc_cli')}",
            "skycalc_endpoint": model.url,
            "skycalc_pwv_mm": f"{pwv}",
            "skycalc_airmass": f"{SKYCALC_AIRMASS}",
            "skycalc_request": "; ".join(
                f"{k}={v}" for k, v in sorted(model.params.items())
            ),
            "skycalc_returned": (
                f"{wave_um.size} samples, 'lam' unit {lam_unit!r}, coverage "
                f"{wave_um[0]:.4f}-{wave_um[-1]:.4f} um"
            ),
        }
    )
    return provenance


def generate_telluric(keys: list[str], cache_dir: Path, command: str) -> None:
    grid, edges = output_grid()

    for key in keys:
        label, pwv, _altitude_m, _native = TELLURIC_SITES[key]
        print(f"[telluric {key}] SkyCalc {label} PWV={pwv} mm")
        model, wave_um, columns, lam_unit = _skycalc_call(
            key, "transmission", pwv, cache_dir
        )

        resampled = resample_flux_conserving(wave_um, columns["trans"], edges)
        n_clipped = int(np.count_nonzero((resampled < 0.0) | (resampled > 1.0)))
        resampled = np.clip(resampled, 0.0, 1.0)

        provenance = _skycalc_provenance(
            key, "transmission", model, wave_um, lam_unit, command
        )
        provenance.update(
            {
                "skycalc_column": (
                    "trans -- the TOTAL line-of-sight transmission, the "
                    "product of molecular absorption, ozone, Rayleigh "
                    "scattering and Mie (aerosol) scattering. It is not "
                    "molecular absorption alone; SkyCalc's trans_ma, trans_o3, "
                    "trans_rs and trans_ms columns are the separate factors"
                ),
                "emission": (
                    "none: this is a transmission product, so airglow, "
                    "atmospheric thermal emission, moonlight, zodiacal light "
                    "and scattered starlight are all switched off in the "
                    "request. The companion sky_radiance_* snapshot carries "
                    "the emission"
                ),
                "reduction": (
                    f"flux-conserving rebin onto the R={R_SAMPLE:.0f} output "
                    f"grid; {n_clipped} samples clipped to [0, 1]"
                ),
            }
        )
        write_csv(
            OUT_DIR / f"telluric_{key}.csv",
            grid,
            resampled,
            "transmission",
            provenance,
        )


def generate_radiance(keys: list[str], cache_dir: Path, command: str) -> None:
    grid, edges = output_grid()

    for key in keys:
        label, pwv, _altitude_m, _native = TELLURIC_SITES[key]
        print(f"[radiance {key}] SkyCalc {label} PWV={pwv} mm")
        model, wave_um, columns, lam_unit = _skycalc_call(
            key, "radiance", pwv, cache_dir
        )

        resampled = resample_flux_conserving(wave_um, columns["flux"], edges)
        n_floored = int(np.count_nonzero(resampled < 0.0))
        resampled = np.maximum(resampled, 0.0)

        provenance = _skycalc_provenance(
            key, "sky_radiance", model, wave_um, lam_unit, command
        )
        provenance.update(
            {
                "skycalc_column": "flux (sky radiance)",
                "units": "ph/s/m2/um/arcsec2 (photon radiance, absolute)",
                "emission": (
                    "airglow lines and continuum plus lower- and upper-"
                    "atmosphere thermal emission. Moonlight, zodiacal light "
                    "and scattered starlight are excluded: they depend on "
                    "pointing and epoch, and nullsim already models zodiacal "
                    "light and site sky brightness elsewhere, so including "
                    "them here would double count"
                ),
                "reduction": (
                    f"flux-conserving rebin onto the R={R_SAMPLE:.0f} output "
                    f"grid; {n_floored} negative samples floored to 0"
                ),
                "note": (
                    "packaged for future spectral sky background work; no "
                    "nullsim stage consumes it yet -- the ground sky is still "
                    "spread as a flat photon-rate density"
                ),
            }
        )
        write_csv(
            OUT_DIR / f"sky_radiance_{key}.csv",
            grid,
            resampled,
            "sky_radiance",
            provenance,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "nullsim_hdc_spectra",
        help="scratch directory for downloaded FITS (never inside the repo)",
    )
    parser.add_argument(
        "--refresh-psg-config",
        action="store_true",
        help=(
            "rebuild the frozen PSG configs under scripts/psg_config/ from the "
            "live service before generating; this changes committed inputs"
        ),
    )
    parser.add_argument(
        "--pin-psg-generator",
        action="store_true",
        help=(
            "expand PSG's GENERATOR defaults into the existing frozen configs "
            "before generating, leaving the harvested atmosphere untouched"
        ),
    )
    sub = parser.add_subparsers(dest="family", required=True)
    sub.add_parser("earth", help="Earth apparent albedo (PSG)")
    sub.add_parser("venus", help="Venus apparent albedo (PSG)")
    stellar = sub.add_parser("stellar", help="G2V/K2V/M2V photon-flux (PHOENIX)")
    stellar.add_argument("--only", choices=sorted(STELLAR_MODELS), action="append")
    telluric = sub.add_parser("telluric", help="site transmission (SkyCalc)")
    telluric.add_argument("--only", choices=sorted(TELLURIC_SITES), action="append")
    radiance = sub.add_parser("radiance", help="site sky radiance (SkyCalc)")
    radiance.add_argument("--only", choices=sorted(TELLURIC_SITES), action="append")
    sub.add_parser("all", help="every family")
    args = parser.parse_args(argv)

    command = f"python scripts/generate_hdc_spectra.py {args.family}"
    cache_dir: Path = args.cache_dir
    set_psg_cache_dir(cache_dir)

    def psg(target: str) -> None:
        if args.refresh_psg_config:
            refresh_psg_config(target)
        if args.pin_psg_generator:
            pin_psg_generator(target)
        generate_albedo(target, command)

    if args.family in ("earth", "venus"):
        psg(args.family)
    elif args.family == "stellar":
        generate_stellar(args.only or sorted(STELLAR_MODELS), cache_dir, command)
    elif args.family == "telluric":
        generate_telluric(args.only or sorted(TELLURIC_SITES), cache_dir, command)
    elif args.family == "radiance":
        generate_radiance(args.only or sorted(TELLURIC_SITES), cache_dir, command)
    else:
        for target in ("earth", "venus"):
            psg(target)
        generate_stellar(sorted(STELLAR_MODELS), cache_dir, command)
        generate_telluric(sorted(TELLURIC_SITES), cache_dir, command)
        generate_radiance(sorted(TELLURIC_SITES), cache_dir, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
