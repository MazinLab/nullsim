---
date: 2026-06-09
---
### Project baselines onto the sky plane for off-zenith pointing

`nullsim.physics.uv_coverage` gains `project_positions_to_sky_plane_m`, which
projects ENU aperture positions onto the plane perpendicular to the line of
sight (axis 0 toward increasing azimuth, axis 1 toward increasing altitude).
Off-zenith, baselines foreshorten as seen from the source, so the planet phase
rows `2*pi/lambda * (x*alpha + y*beta)` are only valid after this projection;
at altitude 90 deg / azimuth 0 it is the identity on (East, North).

Stages that feed `aperture_positions_pupil_m` into sky-angle phase rows now
project at their boundary using `scene.altitude_deg` / `scene.azimuth_deg`
(identity when both are unset, preserving zenith configs):
`sensitivity.detection_curve`, the `chip_optimization.scipy` / `.torch`
kernel and disk rows, `chip_optimization.ideal_kernel` kernel-derivative rows,
and the shared `characterization.spectro` / `characterization.detection_curve`
position reader. The projection lives in the new
`nullsim.stages._utils.project_aperture_positions_for_pointing` boundary
helper so the physics primitives stay state-free. The fringe-observability
piston Jacobians are internal OPD responses, not sky angles, and are left
unprojected. `outputs/plots/null_transmission_map` already projects via its
own richer equatorial HA/Dec path and is unchanged.
