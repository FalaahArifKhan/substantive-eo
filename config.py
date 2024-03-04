from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

SEED = 444

model_specs = {
    "gb-tree":
    {
        "base_model": GradientBoostingClassifier(random_state=111),
        "params": {
                    "n_estimators":[50,250,500]
                    }
    },
    
    "rfc":
    {
        "base_model": RandomForestClassifier(random_state=111),
        "params": {
                    'n_estimators': [50, 250, 500]
                }
    },
    
    "dtc":
    {
        "base_model": DecisionTreeClassifier(random_state=111),
        "params": {
                    'max_depth': [2,5,10],
                    'ccp_alpha': [0.00001, 0.001, 0.01, 0.1, 1, 10],
                    'splitter': ['best', 'random'],
                    'criterion': ['gini', 'entropy']
            
        }
    },
    
    "knn":
    {
        "base_model": KNeighborsClassifier(),
        "params": {
                    'n_neighbors': [2,5,10]
            
        }
    },
    
    "lr":
    {
        "base_model": LogisticRegression(random_state=111, max_iter=200, solver='liblinear'),
        "params": {
                    'C': [0.001, 0.1, 1, 10, 100],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            
        }
    },
    
    "sgd":
    {
        "base_model": SGDClassifier(random_state=111),
        "params": {
                    'eta0': [0.001, 0.01, 0.1]
            
        }
    },
    
    "mlp":
    {
        "base_model": MLPClassifier(hidden_layer_sizes=(100,200,), random_state=111, solver='sgd'),
        "params": {
                    'learning_rate': ['constant', 'invscaling', 'adaptive']
            
        }
    }
    
}