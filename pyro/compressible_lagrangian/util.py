def rp_get(rp, key, default=None):
    try:
        return rp.get_param(key)
    except Exception:
        try:
            return rp.params.get(key, default)
        except Exception:
            if default is None:
                raise
            return default
