"""Tolerance fix for BrickGPT's Gurobi stability score (no original file is modified).

Bug (see investigate_score1/REPORT.md): stability_analysis.py marks a brick as score 1.0 if any force/torque
residual `.X > 0`. Gurobi often returns OPTIMAL with residuals of 1e-31..1e-14 (float noise), so EVERY brick of
a stable structure gets 1.0 (~15% of dataset structures on our machine). We re-exec the original source with
`.X > 0` replaced by `.X > STAB_TOL` and rebind stability_score everywhere it was imported.

Usage: import tolerant_stability; tolerant_stability.patch_all()   (call after the brickgpt / mesh2brick imports)
"""
import importlib
import sys

TOL = 1e-9
_MODULES = [  # (stability module, modules that did `from ... import stability_score`)
    ('brickgpt.stability_analysis.stability_analysis', ['brickgpt.stability_analysis', 'brickgpt.data.brick_structure']),
    ('mesh2brick.stability_analysis.stability_analysis', ['mesh2brick.stability_analysis', 'mesh2brick.data.brick_structure']),
]


def _patch(mod_name: str, users: list[str], tol: float) -> bool:
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            return False
    mod = sys.modules[mod_name]
    src = open(mod.__file__, encoding='utf-8').read()
    n = src.count('.X > 0 or') + src.count('.X > 0\n')
    assert src.count('.X > 0') == 5, 'source changed; check the patch'
    new = src.replace('.X > 0', '.X > STAB_TOL')
    mod.__dict__['STAB_TOL'] = tol
    exec(compile(new, mod.__file__ + ' (tolerant)', 'exec'), mod.__dict__)
    for u in users:
        if u in sys.modules and hasattr(sys.modules[u], 'stability_score'):
            sys.modules[u].stability_score = mod.stability_score
    return n >= 0


def patch_all(tol: float = TOL) -> list[str]:
    return [m for m, users in _MODULES if _patch(m, users, tol)]
