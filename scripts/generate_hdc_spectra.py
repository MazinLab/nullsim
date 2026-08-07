#!/usr/bin/env python
"""Regenerate the packaged HDC template spectra under ``nullsim/data/spectra``.

One-time snapshot generator. The seven CSVs it writes are committed to the
repository and read by `nullsim.physics.spectra.packaged_spectrum_path`; the
test suite never runs this script and never touches the network.

Families and their upstream sources:

``earth``, ``venus``
    NASA Planetary Spectrum Generator (Villanueva et al. 2018, 2022), POSTed to
    ``https://psg.gsfc.nasa.gov/api.php``. Generation reads a *frozen*,
    committed config under ``scripts/psg_config/`` -- never the service's live
    defaults -- and the SHA-256 of that config goes into the CSV header.
    ``--refresh-psg-config`` is the one path that rebuilds those files.

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

``stellar``
    PHOENIX-ACES-AGSS-COND-2011 HiRes FITS (Husser et al. 2013) from
    Goettingen. Downloaded once into ``--cache-dir``, converted from
    erg/s/cm^2/cm to a photon-flux *shape* (multiply by lambda; the absolute
    scale is arbitrary because the consuming stage renormalizes), resampled
    onto a constant-R 300000 grid, degraded from the native R = 500000 to
    R = 30000, then binned onto the output grid.

``telluric``
    ESO SkyCalc (Noll et al. 2012; Jones et al. 2013) via the
    ``skycalc_cli`` REST client. SkyCalc has no Maunakea or Armazones site
    model: both snapshots use its highest site option, ``3060m``, and differ
    only in precipitable water vapour. These are representative site
    snapshots, not exact site models -- the provenance header says so.

All four families are reduced to the same output grid: a constant-resolving-
power log grid at R = 30000 spanning 0.5-1.8 um, produced by
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
R_OUT = 30_000.0

OUT_DIR = REPO_ROOT / "nullsim" / "data" / "spectra"

# Every family must supply source samples at least out to the outermost output
# bin edge, so requests are made with this fractional margin on both ends.
REQUEST_MARGIN = 0.01

# --------------------------------------------------------------------------
# PSG (Earth / Venus disk-integrated albedo at quadrature)
# --------------------------------------------------------------------------

PSG_URL = "https://psg.gsfc.nasa.gov/api.php"
PSG_TIMEOUT_S = 1800

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


PSG_GENERATOR_OVERRIDES = {
    "GENERATOR-RANGEUNIT": "um",
    "GENERATOR-RESOLUTION": "30000",
    "GENERATOR-RESOLUTIONUNIT": "RP",
    "GENERATOR-RESOLUTIONKERNEL": "Y",
    "GENERATOR-RADUNITS": "rif",
    "GENERATOR-TRANS": "N",
    "GENERATOR-LOGRAD": "N",
    "GENERATOR-GAS-MODEL": "Y",
    "GENERATOR-CONT-MODEL": "Y",
    "GENERATOR-TELESCOPE": "SINGLE",
    "GENERATOR-INSTRUMENT": "user",
    "GENERATOR-NOISE": "NONE",
    "GENERATOR-BEAM": "1.0",
    "GENERATOR-BEAM-UNIT": "diffrac",
}

# PUMAS memory limit: 0.5-1.8 um in one request is refused. Empirically Earth
# accepts a ~1.4x wavelength span per call and Venus, whose CO2 line list is
# far denser, only ~1.25x. Log-spaced chunks with a 1% overlap on each seam.
PSG_N_CHUNKS = {"earth": 4, "venus": 7}
PSG_CHUNK_OVERLAP = 0.01

# Chunks are predicted twice in their overlap; if the two predictions disagree
# by more than this (median relative) the stitch is refused rather than quietly
# writing a seam artefact into the template.
PSG_SEAM_TOLERANCE = 1e-3

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
SKYCALC_OBSERVATORY = "3060m"
SKYCALC_AIRMASS = 1.15

TELLURIC_SITES = {
    # key: (label, PWV [mm], real site altitude [m])
    "maunakea": ("Maunakea", 2.5, 4200),
    "armazones": ("Cerro Armazones", 5.0, 3046),
}

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def output_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the R = 30000 output wavelengths and their bin edges."""
    grid = constant_r_grid(LAMBDA_MIN_UM, LAMBDA_MAX_UM, R_OUT)
    half = np.exp(0.5 / R_OUT)
    edges = np.concatenate([grid / half, [grid[-1] * half]])
    return grid, edges


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
        "resolving_power": f"{R_OUT:.0f}",
        "wavelength_range_um": f"{LAMBDA_MIN_UM}-{LAMBDA_MAX_UM}",
        "grid": "constant-R log grid (constant_r_grid)",
        "generator_command": command,
    }


# --------------------------------------------------------------------------
# PSG albedo
# --------------------------------------------------------------------------


def _psg_post(payload: dict[str, str]) -> str:
    response = requests.post(PSG_URL, data=payload, timeout=PSG_TIMEOUT_S)
    response.raise_for_status()
    text = response.text
    errors = [ln for ln in text.splitlines() if "ERROR" in ln.upper()]
    if errors:
        raise RuntimeError("PSG reported an error: " + "; ".join(errors[:3]))
    return text


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
    frame = psg_frame_text(target) + "\n".join(blocks) + "\n"
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
    fractions = _psg_disk_fractions(config)
    print(
        f"  frozen config {path.name} sha256={config_hash[:16]}... "
        f"{fractions.size} disk elements"
    )

    n_chunks = PSG_N_CHUNKS[target]
    bounds = _psg_chunk_edges(n_chunks)
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_chunks):
        lo, hi = bounds[i], bounds[i + 1]
        req_lo = lo * (1.0 - PSG_CHUNK_OVERLAP)
        req_hi = hi * (1.0 + PSG_CHUNK_OVERLAP)
        print(
            f"  PSG: chunk {i + 1}/{n_chunks} {req_lo:.4f}-{req_hi:.4f} um",
            flush=True,
        )
        overrides = dict(PSG_GENERATOR_OVERRIDES)
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

    # Chunks overlap by PSG_CHUNK_OVERLAP on each side of every seam, so each
    # seam has a region predicted twice. Disagreement there is the direct
    # measure of whether stitching is safe; a silent seam artefact would inject
    # a fake spectral feature straight into the CCF template.
    seam_scores: list[float] = []
    for i in range(n_chunks - 1):
        left_w, left_v = raw[i]
        right_w, right_v = raw[i + 1]
        lo = max(left_w[0], right_w[0])
        hi = min(left_w[-1], right_w[-1])
        if hi <= lo:
            raise RuntimeError(f"{target}: chunks {i} and {i + 1} do not overlap")
        probe = left_w[(left_w >= lo) & (left_w <= hi)]
        other = np.interp(probe, right_w, right_v)
        mine = np.interp(probe, left_w, left_v)
        scale = np.maximum(np.abs(mine), np.abs(other))
        good = scale > 0.0
        score = float(np.median(np.abs(mine - other)[good] / scale[good]))
        seam_scores.append(score)
        print(
            f"    seam {i + 1} at {bounds[i + 1]:.4f} um: "
            f"{probe.size} overlapping samples, median rel disagreement "
            f"{score:.3e}"
        )
        if score > PSG_SEAM_TOLERANCE:
            raise RuntimeError(
                f"{target}: seam {i + 1} at {bounds[i + 1]:.4f} um disagrees by "
                f"{score:.3e} > {PSG_SEAM_TOLERANCE:.0e}; refusing to stitch"
            )
    max_seam = max(seam_scores) if seam_scores else 0.0

    pieces = []
    for i, (wave, value) in enumerate(raw):
        keep = np.ones(wave.shape, dtype=bool)
        if i > 0:
            keep &= wave >= bounds[i]
        if i < n_chunks - 1:
            keep &= wave < bounds[i + 1]
        pieces.append((wave[keep], value[keep]))

    wavelength = np.concatenate([p[0] for p in pieces])
    albedo = np.concatenate([p[1] for p in pieces])
    order = np.argsort(wavelength)
    wavelength, albedo = wavelength[order], albedo[order]
    unique = np.concatenate([[True], np.diff(wavelength) > 0.0])
    wavelength, albedo = wavelength[unique], albedo[unique]

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
                "RADUNITS=rif (I/F apparent albedo), RESOLUTION=30000 RP, "
                "RESOLUTIONKERNEL=Y, GAS-MODEL=Y, CONT-MODEL=Y, NOISE=NONE; "
                f"planet column '{target.capitalize()}' (not 'Total', which "
                "includes the star); requested in "
                f"{n_chunks} overlapping log-spaced chunks (PUMAS refuses the "
                "full band in one call) and stitched"
            ),
            "psg_max_seam_disagreement": (
                f"{max_seam:.3e} (median relative, tolerance "
                f"{PSG_SEAM_TOLERANCE:.0e})"
            ),
            "reduction": (
                f"PSG returns the R={R_OUT:.0f} kernel sampled at R=300000; "
                "flux-conserving rebin onto the output grid"
            ),
            "note": (
                "disk-integrated apparent albedo at quadrature phase, not the "
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
        # erg/s/cm^2/cm -> photon-flux shape: F_lambda * lambda / (h c). The
        # constant is dropped; the consuming stage renormalizes.
        photon = flux_all[window] * wave_um

        work = resample_flux_conserving(wave_um, photon, work_edges)
        degraded = convolve_to_resolving_power(
            work_grid, work, R_OUT, r_in=PHOENIX_R_NATIVE
        )
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
                "units": (
                    "arbitrary photon-flux-density shape (F_lambda * lambda, "
                    "h c dropped); the consuming stage renormalizes"
                ),
                "wavelengths": "vacuum, from WAVE_PHOENIX-ACES-AGSS-COND-2011",
                "reduction": (
                    f"flux-conserving rebin onto a constant-R "
                    f"{PHOENIX_R_WORK:.0f} grid, Gaussian LSF degradation "
                    f"R={PHOENIX_R_NATIVE:.0f} -> R={R_OUT:.0f}, "
                    "flux-conserving rebin onto the output grid"
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
# SkyCalc telluric
# --------------------------------------------------------------------------


def generate_telluric(keys: list[str], cache_dir: Path, command: str) -> None:
    from importlib.metadata import version

    from astropy.io import fits
    from skycalc_cli.skycalc import SkyModel

    skycalc_version = version("skycalc_cli")

    grid, edges = output_grid()
    wmin_nm = LAMBDA_MIN_UM * (1.0 - REQUEST_MARGIN) * 1e3
    wmax_nm = LAMBDA_MAX_UM * (1.0 + REQUEST_MARGIN) * 1e3

    for key in keys:
        label, pwv, altitude_m = TELLURIC_SITES[key]
        print(f"[telluric {key}] SkyCalc {label} PWV={pwv} mm")
        model = SkyModel()
        model.stop_on_errors_and_exceptions = False
        model.params.update(
            {
                "airmass": SKYCALC_AIRMASS,
                "pwv_mode": "pwv",
                "pwv": pwv,
                "observatory": SKYCALC_OBSERVATORY,
                "incl_moon": "N",
                "incl_starlight": "N",
                "incl_zodiacal": "N",
                "incl_loweratm": "Y",
                "incl_upperatm": "Y",
                "incl_airglow": "N",
                "incl_therm": "N",
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
        raw = cache_dir / f"skycalc_{key}.fits"
        raw.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(raw))

        with fits.open(raw) as hdul:
            table = hdul[1].data
            lam_unit = str(
                next(c.unit for c in hdul[1].columns if c.name == "lam")
            ).strip()
            lam = np.asarray(table["lam"], dtype=np.float64)
            transmission = np.asarray(table["trans"], dtype=np.float64)

        # skycalc_cli changed the output wavelength unit between v1.3 (um) and
        # v1.4 (nm), so read it from the table rather than assuming, and then
        # confirm the coverage actually matches what was requested.
        scale = {"nm": 1e-3, "um": 1.0, "micron": 1.0}.get(lam_unit)
        if scale is None:
            raise RuntimeError(
                f"SkyCalc returned an unrecognised 'lam' unit {lam_unit!r}"
            )
        wave_um = lam * scale
        if not (
            abs(wave_um[0] - wmin_nm * 1e-3) < 0.02
            and abs(wave_um[-1] - wmax_nm * 1e-3) < 0.02
        ):
            raise RuntimeError(
                f"SkyCalc coverage [{wave_um[0]:.4f}, {wave_um[-1]:.4f}] um "
                f"does not match the requested "
                f"[{wmin_nm * 1e-3:.4f}, {wmax_nm * 1e-3:.4f}] um "
                f"(unit read as {lam_unit!r})"
            )
        print(
            f"  SkyCalc returned {wave_um.size} samples, 'lam' unit "
            f"{lam_unit!r}, coverage {wave_um[0]:.4f}-{wave_um[-1]:.4f} um"
        )

        resampled = resample_flux_conserving(wave_um, transmission, edges)
        n_clipped = int(np.count_nonzero((resampled < 0.0) | (resampled > 1.0)))
        resampled = np.clip(resampled, 0.0, 1.0)

        provenance = base_provenance(
            "transmission",
            "ESO SkyCalc / Cerro Paranal Advanced Sky Model (Noll et al. 2012, "
            "A&A 543, A92; Jones et al. 2013, A&A 560, A91) via skycalc_cli",
            command,
        )
        provenance.update(
            {
                "site": (
                    f"representative of {label} ({altitude_m} m); SkyCalc has "
                    f"no {label} model, so the request uses its highest site "
                    f"option observatory={SKYCALC_OBSERVATORY}"
                ),
                "skycalc_client": f"skycalc_cli {skycalc_version}",
                "skycalc_endpoint": model.url,
                "skycalc_pwv_mm": f"{pwv}",
                "skycalc_request": "; ".join(
                    f"{k}={v}" for k, v in sorted(model.params.items())
                ),
                "skycalc_column": "trans (total atmospheric transmission)",
                "skycalc_returned": (
                    f"{wave_um.size} samples, 'lam' unit {lam_unit!r}, "
                    f"coverage {wave_um[0]:.4f}-{wave_um[-1]:.4f} um"
                ),
                "reduction": (
                    "flux-conserving rebin onto the output grid; "
                    f"{n_clipped} samples clipped to [0, 1]"
                ),
                "note": (
                    "molecular absorption only; airglow, moon, zodiacal and "
                    "starlight emission components excluded"
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
    sub = parser.add_subparsers(dest="family", required=True)
    sub.add_parser("earth", help="Earth apparent albedo (PSG)")
    sub.add_parser("venus", help="Venus apparent albedo (PSG)")
    stellar = sub.add_parser("stellar", help="G2V/K2V/M2V photon-flux (PHOENIX)")
    stellar.add_argument("--only", choices=sorted(STELLAR_MODELS), action="append")
    telluric = sub.add_parser("telluric", help="site transmission (SkyCalc)")
    telluric.add_argument("--only", choices=sorted(TELLURIC_SITES), action="append")
    sub.add_parser("all", help="every family")
    args = parser.parse_args(argv)

    command = f"python scripts/generate_hdc_spectra.py {args.family}"
    cache_dir: Path = args.cache_dir

    if args.family in ("earth", "venus"):
        if args.refresh_psg_config:
            refresh_psg_config(args.family)
        generate_albedo(args.family, command)
    elif args.family == "stellar":
        generate_stellar(args.only or sorted(STELLAR_MODELS), cache_dir, command)
    elif args.family == "telluric":
        generate_telluric(args.only or sorted(TELLURIC_SITES), cache_dir, command)
    else:
        for target in ("earth", "venus"):
            if args.refresh_psg_config:
                refresh_psg_config(target)
            generate_albedo(target, command)
        generate_stellar(sorted(STELLAR_MODELS), cache_dir, command)
        generate_telluric(sorted(TELLURIC_SITES), cache_dir, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
