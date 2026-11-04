"""Instrumented copy of BrickGPT stability_score (original file untouched).
Loads the original source text, injects a diagnostics block before model.close(), and exec's it
inside the brickgpt.stability_analysis package namespace. Diagnostics go to DIAG (module global)."""
import sys, types
from pathlib import Path
sys.path.insert(0, r'D:/HuaweiMoveData/Users/18547/Desktop/icl/pilot')
import common  # noqa: adds 1x3 brick + src path
from common import parse, ROOT
import brickgpt.stability_analysis.stability_analysis as orig
from brickgpt.data import BrickStructure
brick_library = common.bl.brick_library

SRC = Path(orig.__file__).read_text(encoding='utf-8')

INJECT = '''
    _d = dict(status=model.Status, obj=model.objVal, eq_obj=eq_obj.X, mipgap=model.MIPGap,
              objbound=model.ObjBound, bricks=[])
    for key in brick_structure.keys():
        b = brick_structure[key]; bid = str(b["brick_id"])
        if b["ori"] == 0:
            hh = brick_library[bid]["height"]; ww = brick_library[bid]["width"]
        else:
            ww = brick_library[bid]["height"]; hh = brick_library[bid]["width"]
        mc = T_
        for i in range(b["x"], b["x"] + hh):
            for j in range(b["y"], b["y"] + ww):
                fk = gen_key(i, j, b["z"])
                if "f_down" in force_dict[fk]:
                    for k in range(len(force_dict[fk]["f_down"])):
                        mc = min(mc, T_ - force_dict[fk]["f_down"][k].X)
        ix = int(key) - 1
        _d["bricks"].append(dict(key=key, brick_id=bid, x=b["x"], y=b["y"], z=b["z"], h=hh, w=ww,
            fx=force_abs_sum_x[ix].X, fy=force_abs_sum_y[ix].X, fz=force_abs_sum_z[ix].X,
            t1=torque_abs_sum_1[ix].X, t2=torque_abs_sum_2[ix].X, min_c=mc, T=T_))
    DIAG.append(_d)
'''
anchor = '    num_vars = model.NumVars\n'
assert SRC.count(anchor) == 1
PATCHED = SRC.replace(anchor, INJECT + anchor)
# optional extra solver params hook
anchor2 = '    model.setParam("MIPFocus", 1)\n'
assert SRC.count(anchor2) == 1
PATCHED = PATCHED.replace(anchor2, anchor2 + '    for _k, _v in EXTRA_PARAMS.items(): model.setParam(_k, _v)\n')

mod = types.ModuleType('brickgpt.stability_analysis._diag')
mod.__package__ = 'brickgpt.stability_analysis'
mod.DIAG = []
mod.EXTRA_PARAMS = {}
exec(compile(PATCHED, 'stability_analysis_diag', 'exec'), mod.__dict__)


def run(bricks_txt, extra_params=None):
    s = BrickStructure(parse(bricks_txt))
    mod.DIAG.clear()
    mod.EXTRA_PARAMS = dict(extra_params or {})
    scores, *_ = mod.stability_score(s.to_json(), brick_library,
                                     mod.StabilityConfig(world_dimension=(s.world_dim,) * 3))
    d = mod.DIAG[0] if mod.DIAG else None
    return s, scores, d
