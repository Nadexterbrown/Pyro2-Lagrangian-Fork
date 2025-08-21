# compressible_lagrangian (pure Lagrangian 2D solver)
*Pyro2 API-compatible scaffold, corrected to use RuntimeParameters.get_param(name) without defaults.*

- Uses `util.rp_get(rp, key, default)` everywhere a default is desired.
- Purely Lagrangian update (pressure forces/work only; constant cell mass).
- PVRS contact estimate (replaceable with HLLC/CGF later).
