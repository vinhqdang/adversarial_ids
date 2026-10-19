"""Registers FFE-v2 as an attack for run_defenses_v2.py (``--plugin src.defenses.plugin_ffev2``).

The v2 experiment evaluates ``ffe_v2`` only when this module is imported; it needs
``src/attacks/ffe_v2.py`` (class ``FFEv2Attack``) and only uses ``predict_proba`` and
``get_split_thresholds`` of the target, which every defended model in this package provides.
"""

from src.attacks.ffe_v2 import FFEv2Attack
from src.defenses.attackers import register_attack

register_attack('ffe_v2', lambda model, ctx: FFEv2Attack(model, ctx.schema, feature_stds=ctx.stds))
