from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

from sklearn import linear_model
from sklearn import metrics

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
        

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features*100)
            )
        else:
            raise Exception("Invalid type of feature n_features")
            
    def fit(self, X, y):
        return self.selection.fit(X, y)

    def transform(self, X):
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)


class GreedyFeatureSelection:
    
    def __init__(self, max_loop = 100):
        self.max_loop = max_loop

    def evaluate_score(self, X, y):
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        loop = 0
        while loop < self.max_loop:
            loop += 1
            this_feature = None
            best_score = 0
            for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)
                if score > best_score:
                    this_feature = feature
                    best_score = score
                if this_feature is not None:
                    good_features.append(this_feature)
                    best_scores.append(best_score)
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        return best_scores[:-1], good_features[:-1]
    
    def __call__(self, X, y):
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores