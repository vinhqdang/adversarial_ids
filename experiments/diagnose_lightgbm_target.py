"""Causal test: change FFE's TCP-window target (64240 default) to a value inside the training support (65535)."""
import os, sys, types, collections
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import load_arrays, detected_by_all, XGB_CFG
from src.datasets.schema import get_cicids_schema
from src.models.tree_models import NIDSEnsembleTrainer
import src.attacks.flow_field_evasion as ffe_mod
import src.attacks.base  # noqa

def ffe_with_target(target):
    """FFE module with the window target literal replaced (source patch, src/ untouched)."""
    src = open(ffe_mod.__file__).read().replace('64240.0', f'{float(target)}')
    mod = types.ModuleType('src.attacks.ffe_patched'); mod.__package__ = 'src.attacks'
    mod.__file__ = ffe_mod.__file__; exec(compile(src, ffe_mod.__file__, 'exec'), mod.__dict__)
    return mod.FlowFieldEvasionAttack

SEED=42; schema=get_cicids_schema(); d=load_arrays('cic2017')
X_tr,y_tr,_=d['train']; X_te,y_te,fam=d['test']; stds=X_tr.std(0)+1e-6
def fit(cfg,t):
    m=NIDSEnsembleTrainer(t,cfg); m.train(X_tr,y_tr); return m
xgb=fit(XGB_CFG,'xgboost'); xgb.model.get_booster().set_param({'device':'cpu'})
lg=fit({'n_estimators':60,'max_depth':6,'learning_rate':0.1,'random_state':SEED},'lightgbm')
mal=np.where(y_te==1)[0]; det=mal[detected_by_all([xgb,lg],X_te[mal])]
idx=det[np.random.RandomState(SEED).choice(len(det),1000,replace=False)]; Xc=X_te[idx]; fc=fam[idx]
for target in (64240, 65535):
    A=ffe_with_target(target)
    for nm,m in [('lightgbm',lg),('xgboost',xgb)]:
        atk=A(m,schema,step_budget=20,feature_stds=stds); res=[atk.attack_sample(x) for x in Xc]
        ok=np.array([r.success for r in res]); q=np.mean([r.queries_used for r in res])
        print(f'target={target} {nm}: ASR {100*ok.mean():.1f}  avg queries {q:.1f}  by family',
              {f:round(100*ok[fc==f].mean(),1) for f in sorted(set(fc))}, flush=True)
