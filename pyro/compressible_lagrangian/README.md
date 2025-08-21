# compressible_lagrangian (API fix)
*Now fully aligned with Pyro2 problem API: `init_data(my_data, rp)`*
- `Simulation.initialize()` calls `problem_func(my_data, rp)` where `my_data` is a cc_data-like shim.
- After init, converts to Lagrangian mass via `m = rho * area`.
- Uses `rp_get()` helper for params where defaults are desirable.
