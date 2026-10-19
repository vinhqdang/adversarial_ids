"""Inspect init_fwd_win_byts handling in LightGBM vs XGBoost on CIC-IDS2017 DoS flows."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import load_arrays, detected_by_all, XGB_CFG
from src.models.tree_models import NIDSEnsembleTrainer
SEED=42; d=load_arrays('cic2017'); X_tr,y_tr,_=d['train']; X_te,y_te,fam=d['test']
LGB={'n_estimators':60,'max_depth':6,'learning_rate':0.1,'random_state':SEED}
def fit(cfg,t='lightgbm',**kw):
    m=NIDSEnsembleTrainer(t,cfg); m.train(X_tr,y_tr); return m
xgb=fit(XGB_CFG,'xgboost'); xgb.model.get_booster().set_param({'device':'cpu'}); lg=fit(LGB)
w=X_tr[:,7]
for lab in (0,1):
    v,c=np.unique(w[y_tr==lab],return_counts=True); o=np.argsort(-c)[:6]; print('label',lab,'top win values',list(zip(v[o],(100*c[o]/c.sum()).round(1))))
for nm,m in [('xgb',xgb),('lgb',lg)]:
    th=sorted(set(round(t,1) for j,t in m.get_split_thresholds() if j==7)); print(nm,'win thresholds',len(th),[t for t in th if 1000<t<70000][:40])
mal=np.where(y_te==1)[0]; det=mal[detected_by_all([xgb,lg],X_te[mal])]
Z=X_te[det][fam[det]=='DoS']; print('DoS win values',np.unique(Z[:,7],return_counts=True))
for val in [0,1024,8192,14600,29200,45000,60000,64240,65535]:
    A=Z.copy(); A[:,7]=val; print(val,'benign% xgb',round(100*(xgb.predict_proba(A)[:,1]<.5).mean(),1),'lgb',round(100*(lg.predict_proba(A)[:,1]<.5).mean(),1))
# LightGBM variants: how does lgb respond to win=64240 under other training setups
for nm,extra in [('default num_leaves31',{}),('num_leaves=64',{'num_leaves':64}),('max_bin=1023',{'max_bin':1023}),('min_child_samples=5',{'min_child_samples':5}),('n_est=300',{'n_estimators':300}),('lambda_l2=0 min_data_in_bin=1',{'min_data_in_bin':1,'max_bin':4095})]:
    m=fit({**LGB,**extra}); ok=detected_by_all([m],Z); A=Z[ok].copy(); A[:,7]=64240
    print(nm,'detected',int(ok.sum()),'win=64240 benign%',round(100*(m.predict_proba(A)[:,1]<.5).mean(),1))
