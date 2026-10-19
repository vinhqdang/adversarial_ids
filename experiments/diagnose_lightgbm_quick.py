"""Quick structural diagnosis of FFE vs LightGBM on CIC-IDS2017 (200-flow cohort)."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import load_arrays, detected_by_all, XGB_CFG
from src.datasets.full_data import FEATURES
from src.datasets.schema import get_cicids_schema
from src.models.tree_models import NIDSEnsembleTrainer
from src.attacks.flow_field_evasion import FlowFieldEvasionAttack
import lightgbm as lgb
SEED=42; schema=get_cicids_schema(); d=load_arrays('cic2017')
X_tr,y_tr,_=d['train']; X_te,y_te,fam=d['test']; stds=X_tr.std(0)+1e-6
LGB={'n_estimators':60,'max_depth':6,'learning_rate':0.1,'random_state':SEED}
def fit(cfg,t='lightgbm'):
    m=NIDSEnsembleTrainer(t,cfg); m.train(X_tr,y_tr); return m
xgb=fit(XGB_CFG,'xgboost'); xgb.model.get_booster().set_param({'device':'cpu'}); lg=fit(LGB)
mal=np.where(y_te==1)[0]; det=mal[detected_by_all([xgb,lg],X_te[mal])]
idx=det[np.random.RandomState(SEED).choice(len(det),1000,replace=False)][:200]
Xc=X_te[idx]; fc=fam[idx]
print('cohort fams',collections.Counter(fc),flush=True)
bst=lg.model.booster_; g=bst.feature_importance('gain'); s=bst.feature_importance('split')
for j,f in enumerate(FEATURES): print(f'{f:22s} gain {g[j]/g.sum():.3f} splits {s[j]}')
xg=xgb.model.get_booster().get_score(importance_type='total_gain'); t=sum(xg.values())
print('XGB gain',{FEATURES[int(k[1:])]:round(v/t,3) for k,v in xg.items()})
p0=lg.predict_proba(Xc)[:,1]; px=xgb.predict_proba(Xc)[:,1]
print('start score LGB q',np.quantile(p0,[.1,.5,.9]).round(3),'XGB',np.quantile(px,[.1,.5,.9]).round(3))
atk=FlowFieldEvasionAttack(lg,schema,step_budget=20,feature_stds=stds)
res=[atk.attack_sample(x) for x in Xc]; ok=np.array([r.success for r in res]); print('LGB ASR',100*ok.mean(),flush=True)
for f in sorted(set(fc)): print(f, (fc==f).sum(), round(100*ok[fc==f].mean(),1))
pe=lg.predict_proba(np.array([r.x_adversarial for r in res]))[:,1]
print('fail start',p0[~ok].mean(),'succ start',p0[ok].mean(),'fail end q',np.quantile(pe[~ok],[.1,.5,.9]).round(3))
chg=np.array([np.abs(r.x_adversarial-r.x_original)>0 for r in res]); print('changed feats (fail)',dict(zip(FEATURES,chg[~ok].mean(0).round(2))))
# root splits & first-tree
dm=bst.dump_model()
roots=collections.Counter(FEATURES[t['tree_structure']['split_feature']] for t in dm['tree_info'] if 'split_feature' in t['tree_structure'])
print('LGB root feats',roots)
# failing flow: feature values of the few most important unreachable features
fl=np.where(~ok)[0][:5]
for i in fl: print(fc[i],dict(zip(FEATURES,Xc[i].round(1))),'p0',round(p0[i],3),'pend',round(pe[i],3))
