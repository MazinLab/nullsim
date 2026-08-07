#!/usr/bin/env python
"""Regenerate the packaged HDC template spectra under ``nullsim/data/spectra``.

One-time snapshot generator. The seven CSVs it writes are committed to the
repository and read by `nullsim.physics.spectra.packaged_spectrum_path`; the
test suite never runs this script and never touches the network.

Families and their upstream sources:

``earth``, ``venus``
    NASA Planetary Spectrum Generator (Villanueva et al. 2018, 2022), POSTed to
    ``https://psg.gsfc.nasa.gov/api.php``. PSG expands a pinned minimal config
    into a full atmosphere (MERRA-2 for Earth on a fixed date, PSG's Venus
    template), we override the viewing geometry to quadrature and the generator
    block to ``rif`` units, and read the ``Total`` column, which PSG documents
    as "I/F [apparent albedo]". PUMAS refuses 0.5-1.8 um in one shot on memory
    grounds, so the band is requested in overlapping log-spaced chunks and
    stitched.

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
# PSG (Earth / Venus geometric albedo)
# --------------------------------------------------------------------------

PSG_URL = "https://psg.gsfc.nasa.gov/api.php"
PSG_TIMEOUT_S = 1800

# Fixed epoch: PSG pulls the Earth atmosphere from the NASA/MERRA-2 archive for
# this date, so pinning it pins the atmospheric state.
PSG_DATE = "2026/03/20 12:00"

PSG_MINIMAL_CFG = {
    "earth": (
        "<OBJECT>Planet\n"
        "<OBJECT-NAME>Earth\n"
        f"<OBJECT-DATE>{PSG_DATE}\n"
        "<GEOMETRY>Observatory\n"
    ),
    "venus": (
        "<OBJECT>Planet\n"
        "<OBJECT-NAME>Venus\n"
        f"<OBJECT-DATE>{PSG_DATE}\n"
        "<GEOMETRY>Observatory\n"
    ),
}

# Quadrature: cos(phase) = cos(i)cos(e) + sin(i)sin(e)cos(azimuth) vanishes for
# i = 60 deg, e = 30 deg, azimuth = 180 deg, so the phase angle is exactly 90
# deg. The pair also gives a sane disk-average airmass, sec(60) + sec(30).
PSG_GEOMETRY_OVERRIDES = {
    "GEOMETRY-SOLAR-ANGLE": "60.000",
    "GEOMETRY-OBS-ANGLE": "30.000",
    "GEOMETRY-AZIMUTH": "180.000",
}

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


def _psg_expand_config(target: str) -> str:
    """Ask PSG to fill in ephemeris, geometry, and atmosphere for the target."""
    print(f"  PSG: expanding minimal config for {target}")
    return _psg_post(
        {
            "type": "cfg",
            "wephm": "y",
            "watm": "y",
            "file": PSG_MINIMAL_CFG[target],
        }
    )


def _psg_apply_overrides(config: str, overrides: dict[str, str]) -> str:
    kept = [
        line
        for line in config.splitlines()
        if not any(line.startswith(f"<{key}>") for key in overrides)
    ]
    kept.extend(f"<{key}>{value}" for key, value in overrides.items())
    return "\n".join(kept) + "\n"


def _psg_parse_spectrum(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull the wavelength and ``Total`` columns out of a PSG rad file."""
    columns: list[str] | None = None
    waves: list[float] = []
    values: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if body.startswith("Wave"):
                columns = body.split()
            continue
        if not stripped:
            continue
        fields = stripped.split()
        waves.append(float(fields[0]))
        values.append(float(fields[columns.index("Total") if columns else 1]))
    if not waves:
        raise RuntimeError("PSG returned no spectral rows")
    return np.asarray(waves, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _psg_chunk_edges(n_chunks: int) -> np.ndarray:
    lo = LAMBDA_MIN_UM * (1.0 - REQUEST_MARGIN)
    hi = LAMBDA_MAX_UM * (1.0 + REQUEST_MARGIN)
    return lo * (hi / lo) ** (np.arange(n_chunks + 1) / n_chunks)


def generate_albedo(target: str, command: str) -> None:
    print(f"[{target}] PSG apparent albedo at quadrature")
    config = _psg_expand_config(target)
    config = _psg_apply_overrides(config, PSG_GEOMETRY_OVERRIDES)

    n_chunks = PSG_N_CHUNKS[target]
    bounds = _psg_chunk_edges(n_chunks)
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_chunks):
        lo, hi = bounds[i], bounds[i + 1]
        req_lo = lo * (1.0 - PSG_CHUNK_OVERLAP)
        req_hi = hi * (1.0 + PSG_CHUNK_OVERLAP)
        print(f"  PSG: chunk {i + 1}/{n_chunks} {req_lo:.4f}-{req_hi:.4f} um")
        overrides = dict(PSG_GENERATOR_OVERRIDES)
        overrides["GENERATOR-RANGE1"] = f"{req_lo:.6f}"
        overrides["GENERATOR-RANGE2"] = f"{req_hi:.6f}"
        chunk_cfg = _psg_apply_overrides(config, overrides)
        text = _psg_post(
            {"type": "rad", "wephm": "n", "wgeo": "n", "file": chunk_cfg}
        )
        wave, value = _psg_parse_spectrum(text)
        # Trim the overlap back to the nominal seam, keeping the first chunk's
        # blue margin and the last chunk's red margin.
        keep = np.ones(wave.shape, dtype=bool)
        if i > 0:
            keep &= wave >= lo
        if i < n_chunks - 1:
            keep &= wave < hi
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
    resampled = np.clip(resampled, 0.0, 1.0)

    provenance = base_provenance(
        "albedo",
        "NASA Planetary Spectrum Generator (Villanueva et al. 2018, 2022), "
        f"api.php, object {target.capitalize()}",
        command,
    )
    provenance.update(
        {
            "psg_object_date": PSG_DATE,
            "psg_geometry": (
                "Observatory; incidence 60 deg, emission 30 deg, azimuth 180 "
                "deg => phase angle 90 deg (quadrature), disk-representative"
            ),
            "psg_generator": (
                "RADUNITS=rif (I/F apparent albedo), RESOLUTION=30000 RP, "
                "RESOLUTIONKERNEL=Y, GAS-MODEL=Y, CONT-MODEL=Y, NOISE=NONE; "
                f"requested in {n_chunks} overlapping log-spaced chunks "
                "(PUMAS refuses the full band in one call) and stitched"
            ),
            "reduction": (
                f"PSG returns the R={R_OUT:.0f} kernel sampled at R=300000; "
                "flux-conserving rebin onto the output grid"
            ),
            "note": (
                "apparent albedo at quadrature phase, not the zero-phase "
                f"geometric albedo; {n_clipped} samples clipped to [0, 1]"
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
    from astropy.io import fits
    from skycalc_cli.skycalc import SkyModel

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
            wave_um = np.asarray(table["lam"], dtype=np.float64) * 1e-3
            transmission = np.asarray(table["trans"], dtype=np.float64)

        resampled = resample_flux_conserving(wave_um, transmission, edges)
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
                "skycalc_request": (
                    f"airmass={SKYCALC_AIRMASS}, pwv_mode=pwv, pwv={pwv} mm, "
                    f"vacair=vac, wgrid_mode=fixed_spectral_resolution, "
                    f"wres={SKYCALC_R_SAMPLE}, lsf_type=Gaussian, "
                    f"lsf_gauss_fwhm={SKYCALC_LSF_FWHM_BINS} bins "
                    f"(=> R={R_OUT:.0f}), column 'trans'"
                ),
                "reduction": (
                    "flux-conserving rebin onto the output grid, clipped to "
                    "[0, 1]"
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
        generate_albedo(args.family, command)
    elif args.family == "stellar":
        generate_stellar(args.only or sorted(STELLAR_MODELS), cache_dir, command)
    elif args.family == "telluric":
        generate_telluric(args.only or sorted(TELLURIC_SITES), cache_dir, command)
    else:
        generate_albedo("earth", command)
        generate_albedo("venus", command)
        generate_stellar(sorted(STELLAR_MODELS), cache_dir, command)
        generate_telluric(sorted(TELLURIC_SITES), cache_dir, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
