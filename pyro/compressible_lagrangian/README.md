
# compressible_lagrangian_pure (Pyro2 plugin)

A purely Lagrangian (no ALE, no remap) 2-D compressible Euler solver
for Pyro2 on a moving, deforming quad mesh. Face speeds come from a
normal 1-D HLLC Riemann solve; per-cell mass is constant; density is
updated from mass/area after node motion.

## Key points
- No Eulerian fluxes and no remap anywhere.
- Face velocity = contact speed u* from HLLC along the face normal.
- Pressure forces and pressure work drive momentum/energy.
- Optional VN-R artificial viscosity and hourglass control (off by default).
- Pyro2 API: package exposes `simulation` submodule and `Simulation` class.

## Usage
1. Add `"compressible_lagrangian_pure"` to `valid_solvers` in `pyro/pyro_sim.py`.
2. Ensure this package is importable as `pyro.compressible_lagrangian_pure`.
3. Run (example):
   ```bash
   python -m pyro.pyro_sim compressible_lagrangian_pure noh2d pyro/compressible_lagrangian_pure/_defaults/inputs.noh
   ```

## Files
- `simulation.py`: Pyro-compatible driver and stepping loop (SSP-RK2).
- `mesh.py`: structured quad moving mesh; node motion by averaged face speeds.
- `state.py`: per-cell mass m, velocity u, specific energy E, density from m/area.
- `riemann.py`: normal HLLC returning u* and p*.
- `forces.py`: pressure forces/work accumulation.
- `viscosity.py`: VN-R q and placeholder hourglass control.
- `boundary.py`: moving wall (piston), outflow, periodic y.
- `problems/`: noh2d, sedov2d, sod2d_channel, piston2d.
- `_defaults/`: example input decks.

## Notes
This is a compact but complete scaffold. For production:
- Implement full MUSCL-Hancock with Green-Gauss gradients on deforming quads.
- Improve node velocity recovery via least-squares from adjacent face speeds.
- Add robust hourglass control (e.g., Flanagan-Belytschko).
- Validate on Noh, Sedov, Sod, and piston channel with regression tests.
