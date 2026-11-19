# PIP in OnShape (parametric)

Fast path: `python cad/pip_cad.py` -> `cad/out/*.step`. In OnShape: Create tab -> Import -> `PIP_assembly.step` (or each part STEP into one Part Studio).
Native-parametric path: build the Part Studio from the variables and hardpoints below, so changing a number regenerates the neck.

## Variable Studio (units mm / deg) — identical to pip/params.json
| var | value | var | value |
|---|---|---|---|
| #base_r | 45 | #plat_r | 38 |
| #base_gap | 28 deg | #plat_gap | 32 deg |
| #horn | 18 | #rod | 50 |
| #h0 | `sqrt(#rod^2 - d^2)` = 48.51 (from hardpoints.csv) | #plate_t | 4 |
| #body_r | 62 | #body_h | 60 |
| #servo | 12 x 23 x 24 | #cam | 8.5 x 8.5 |
| #mic_d / #mic_pitch | 1.6 / 30 | #spk_d | 40 |
| #ant_len / #ant_pitch | 60 / 50 | | |

## Feature order
1. Sketch "Base joints" on Top plane: circle R=#base_r; points at 90/210/330 deg +-#base_gap/2 (`hardpoints.csv`, rows base_joint).
2. Extrude stewart_base plate; cut 6 servo pockets (#servo) rotated to the horn direction (beta = joint angle +-90 deg, alternating).
3. Sketch "Platform joints": circle R=#plat_r; points at 30/150/270 deg +-#plat_gap/2, offset up by #h0 (rows platform_joint).
4. Extrude top plate: 6 ball-stud holes, camera window at front (+Y), 2 mic holes at +-#mic_pitch/2, 2 antenna bosses at +-#ant_pitch/2.
5. Horns (#horn shaft-to-stud) and rods (#rod, ball-socket ends); mate each rod from horn tip to platform joint (Ball mates).
6. Body: base shell (speaker bay front, 6x grille holes, rear cable slot) -> turntable with bearing seat + yaw servo pocket -> Revolute mate on Z (limit +-60 deg).
7. Antennas: 2 micro-servos in the head, Revolute mates (+-80 deg).
Verify: motion-study the top plate +-18 deg roll/pitch, +-30 deg yaw; servo angles must stay inside +-70 deg (compare with `StewartNeck.ik`).

NOTE: dimensions are a Reachy-Mini-*style* starting point, not Pollen's exact geometry. Pull their open CAD/URDF, then edit params.json (or the Variable Studio) to match their servo/horn/rod values; the sim follows automatically.
