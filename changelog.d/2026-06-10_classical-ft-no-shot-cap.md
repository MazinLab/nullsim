---
date: 2026-06-10
---
### Classical fringe tracker: shot noise no longer capped at the open-loop budget

The KILO-heritage cap of the full classical residual at the open-loop AO
total hid the sensor shot noise once it exceeded the atmospheric budget
(between J=8 and J=10): the published residual understated J=12 by ~7x in
rms and the sample-rate optimizer's iid term went rate-independent, pinning
the bandwidth at sample_rate_max while still claiming full common-mode
rejection. The raw lag + shot + uncorrectable + irreducible sum is now
published; the numeric optimizer trades shot noise against disturbance
rejection honestly (a starved loop slows down), and the diagnostic flag is
renamed shot_exceeds_open_loop. The photon-Kalman branch was never
affected (its shot noise enters through the uncapped CRLB path).
