"""Which single FFE moves flip DoS flows for XGBoost but not LightGBM? Also capacity ablation (fast, vectorised)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import load_arrays, detected_by_all, XGB_CFG
from src.datasets.full_data import FEATURES
from src.models.tree_models import NIDSEnsembleTrainer
SEED=42; d=load_arrays('cic2017'); X_tr,y_tr,fam_tr=d['train']; X_te,y_te,fam=d['test']
LGB={'n_estimators':60,'max_depth':6,'learning_rate':0.1,'random_state':SEED}
def fit(cfg,t='lightgbm'):
    m=NIDSEnsembleTrainer(t,cfg); m.train(X_tr,y_tr); return m
xgb=fit(XGB_CFG,'xgboost'); xgb.model.get_booster().set_param({'device':'cpu'}); lg=fit(LGB)
mal=np.where(y_te==1)[0]; det=mal[detected_by_all([xgb,lg],X_te[mal])]
X=X_te[det]; f=fam[det]
print('detected cohort by family',{k:int((f==k).sum()) for k in set(f)})
def rate(m,Z): return (m.predict_proba(Z)[:,1]>=0.5).mean()*100
for fam_name in ['DoS','DDoS','PortScan']:
    Z=X[f==fam_name]
    for nm,m in [('xgb',xgb),('lgb',lg)]:
        print(fam_name,nm,'N',len(Z))
        # window morph
        W=Z.copy(); W[:,7]=64240; print('  win=64240 -> benign %',100-rate(m,W))
        # extreme duration dilation (rates recomputed)
        for mult in [10,1000]:
            T=Z.copy(); T[:,6]=np.minimum(Z[:,6]*mult,1.1e8); T[:,8]=1e6*(T[:,4]+T[:,5])/T[:,6]; T[:,9]=1e6*(T[:,2]+T[:,3])/T[:,6]
            print(f'  duration x{mult} -> benign %',100-rate(m,T))
        # combined win + dilation
        C=W.copy(); C[:,6]=np.minimum(Z[:,6]*1000,1.1e8); C[:,8]=1e6*(C[:,4]+C[:,5])/C[:,6]; C[:,9]=1e6*(C[:,2]+C[:,3])/C[:,6]
        print('  win+dur x1000 -> benign %',100-rate(m,C))
# immutable-feature oracle: set tot_bwd/totlen_bwd (immutable) to benign medians
ben=X_tr[y_tr==0]; med=np.median(ben,0)
for fam_name in ['DoS','DDoS']:
    Z=X[f==fam_name]
    for nm,m in [('xgb',xgb),('lgb',lg)]:
        for cols in [[3,5],[0],[7]]:
            O=Z.copy(); O[:,cols]=med[cols]; print(fam_name,nm,'replace',[FEATURES[c] for c in cols],'by benign median -> benign %',100-rate(m,O))
