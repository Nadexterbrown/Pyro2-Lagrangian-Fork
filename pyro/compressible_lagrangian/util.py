def rp_get(rp, key, default=None):
    """
    Pyro2 RuntimeParameters shim:
    - Try rp.get_param(key) (Pyro2 API: 1 positional arg)
    - Else try rp.params dict
    - Else return default (or raise if default is None)
    """
    try:
        return rp.get_param(key)
    except Exception:
        try:
            return rp.params.get(key, default)
        except Exception:
            if default is None:
                raise
            return default
