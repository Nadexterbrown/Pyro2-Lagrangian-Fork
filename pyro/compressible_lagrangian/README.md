# compressible_lagrangian_pure

A **purely Lagrangian (no ALE, no remap)** 2‑D compressible Euler solver package
that plugs into **Pyro2** with an API compatible with the standard `compressible` solver.

## Status
This is a **functional scaffold** implementing the required API, moving mesh,
PVRS contact estimates, pressure‑force/work updates, and SSP‑RK2 stepping.
Stabilization (vNR) exists but is **off by default**. Hourglass control hooks
are left for future work.

## Using with `pyro_sim.py`
1. Add `"compressible_lagrangian_pure"` to `valid_solvers` in `pyro/pyro_sim.py`.
2. Ensure Python can import this package as `pyro.compressible_lagrangian_pure`.
3. Run e.g.
   ```bash
   python -m pyro.pyro_sim compressible_lagrangian_pure piston2d inputs.piston
   ```

## Design
* **Per‑cell mass** `m` is constant; density comes from `m/area` after node motion.
* **Face star speed/pressure** from a face‑normal **Riemann** (PVRS approx here) drive
  pressure forces and pressure work only; **no Eulerian mass flux** is used anywhere.
* **Nodes move** with a velocity assembled from face star speeds; geometry is recomputed
  each stage to satisfy geometric conservation at first order.

## Files
See the source tree for `simulation.py` (driver), `mesh.py`, `state.py`, `riemann.py`,
`reconstruction.py`, `time_integration.py`, `forces.py`, `viscosity.py`, `boundary.py`,
and problems/tests.

## TODO
- Replace PVRS with exact or HLLC normal Riemann in the material frame.
- General tangential treatment / hourglass control.
- Richer boundary conditions and inputs files.
