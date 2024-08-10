from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

model_map = {'rfc': RandomForestClassifier,
             'lr': LogisticRegression,
             'xgbc': XGBClassifier,
             'mnb': MultinomialNB}
