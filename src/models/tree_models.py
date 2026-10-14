"""
Model training and tree ensemble split extraction for XGBoost, LightGBM, and CatBoost.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb


class NIDSEnsembleTrainer:
    """
    Trains and benchmarks GBDT models on NIDS data, and exposes
    internal tree split structures for adversarial partition analysis.
    """
    def __init__(self, model_type: str = 'xgboost', params: Dict[str, Any] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self._init_model()

    def _init_model(self):
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'eval_metric': 'logloss',
                'random_state': 42
            }
            default_params.update(self.params)
            self.model = XGBClassifier(**default_params)
            
        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': -1
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            
        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                default_params = {
                    'iterations': 50,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'verbose': False,
                    'random_seed': 42
                }
                default_params.update(self.params)
                self.model = CatBoostClassifier(**default_params)
            except ImportError:
                raise ImportError("CatBoost is not yet installed. Please wait for pip installation.")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probs))
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def get_split_thresholds(self, feature_names: List[str] = None) -> List[Tuple[int, float]]:
        splits = []
        try:
            if self.model_type == 'xgboost':
                booster = self.model.get_booster()
                dump = booster.get_dump(dump_format='text')
                for tree_text in dump:
                    for line in tree_text.splitlines():
                        line = line.strip()
                        if '<' in line and '[' in line and ']' in line:
                            bracket_content = line[line.find('[') + 1:line.find(']')]
                            parts = bracket_content.split('<')
                            if len(parts) == 2:
                                feat_str = parts[0].strip()
                                try:
                                    if feat_str.startswith('f') and feat_str[1:].isdigit():
                                        f_idx = int(feat_str[1:])
                                    elif feature_names and feat_str in feature_names:
                                        f_idx = feature_names.index(feat_str)
                                    else:
                                        continue
                                    thresh = float(parts[1].strip())
                                    splits.append((f_idx, thresh))
                                except ValueError:
                                    pass
            elif self.model_type == 'lightgbm':
                dump = self.model.booster_.dump_model()
                def recurse_tree(node):
                    if 'split_feature' in node and 'threshold' in node:
                        feat = node['split_feature']
                        thresh = float(node['threshold'])
                        if isinstance(feat, int):
                            splits.append((feat, thresh))
                        elif isinstance(feat, str):
                            if feat.startswith('Column_') and feat[7:].isdigit():
                                splits.append((int(feat[7:]), thresh))
                            elif feature_names and feat in feature_names:
                                splits.append((feature_names.index(feat), thresh))
                        if 'left_child' in node:
                            recurse_tree(node['left_child'])
                        if 'right_child' in node:
                            recurse_tree(node['right_child'])
                for tree_info in dump.get('tree_info', []):
                    if 'tree_structure' in tree_info:
                        recurse_tree(tree_info['tree_structure'])
        except Exception:
            pass
        return splits
