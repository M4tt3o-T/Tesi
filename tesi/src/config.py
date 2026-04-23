"""
Modulo di configurazione globale per gli esperimenti di classificazione.

Contiene la definizione dei percorsi fisici, dei parametri base, delle liste
dei componenti chimici e dei dizionari contenenti le istanze dei modelli
di Machine Learning utilizzati nelle varie suite di test.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Union

# Importiamo la classe base dei classificatori per il type checking
from sklearn.base import ClassifierMixin

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================================
# PERCORSI DI SISTEMA
# ==========================================
BASE_DIR: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..") + os.sep
TRAIN_DIR: str = os.path.join(BASE_DIR, "TRAINING", "NORMALIZED") + os.sep
CACHE_FILE: str = os.path.join(BASE_DIR, "EXTRACTED_DATA.npy")

# ==========================================
# PARAMETRI DATI E SOGLIE
# ==========================================
SENSORS: np.ndarray = np.arange(16)
NMIN: int = 4
SOGLIA: float = 0.25

# ==========================================
# LISTE COMPONENTI CHIMICI
# ==========================================
COMPONENTS: List[str] = [
    "ACETIC_ACID",
    "AIR",
    "ACETONE",
    "AMMONIUM_CHLORIDE",
    "AMMONIA",
    "CALCIUM_NITRATE",
    "BUTANE",
    "BIOETHANOL",
    "APPLE_VINEGAR",
    "DIESEL",
    "GASOLINE",
    "FORMIC_ACID",
    "ETHANOL",
    "ISOPROPANOL",
    "HYDROGEN_PEROXIDE",
    "METHANE",
    "LIGHTER_FLUID",
    "KEROSENE",
    "RED_WINE",
    "PHOSPHORIC_ACID",
    "NITROMETHANE",
    "WATER_VAPOR",
    "UREA",
    "SODIUM_HYDROXIDE",
    "BALSAMIC_VINEGAR",
]

# Composti che hanno un trattamento speciale nello split "VECCHIO/NUOVO"
LISTA_AMMESSI_STAT: List[str] = ["CALCIUM_NITRATE", "ISOPROPANOL"]

# Gruppi usati per la classificazione gerarchica (Stage 2)
HYDROCARBONS: List[str] = [
    "METHANE",
    "BUTANE",
    "GASOLINE",
    "LIGHTER_FLUID",
    "DIESEL",
    "KEROSENE",
]

# ==========================================
# MODELLI DI DEFAULT (SUITE CLASSICHE)
# ==========================================
SLOW_CLASSIFIERS: Dict[str, ClassifierMixin] = {
    "SVM (gaussian)": SVC(kernel="rbf", random_state=42, verbose=False, max_iter=5000),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM (Linear Kernel)": SVC(C=10, kernel="linear", probability=False),
}

FAST_CLASSIFIERS: Dict[str, Union[ClassifierMixin, LGBMClassifier]] = {
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=100),
    "Ridge Classifier": RidgeClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Linear SVC": LinearSVC(C=10, dual=False, random_state=42, max_iter=1000),
    "Linear Discriminant": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant": QuadraticDiscriminantAnalysis(reg_param=0.01),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist"
    ),
    "LightGBM": LGBMClassifier(verbose=-1, random_state=42, n_jobs=-1),
}

DIFFERENT_MLPS: Dict[str, ClassifierMixin] = {
    "MLP Originale (20,)": MLPClassifier(
        hidden_layer_sizes=(20,),
        activation="relu",
        solver="sgd",
        alpha=1.0e-2,
        max_iter=5000,
        tol=1.0e-3,
        n_iter_no_change=100,
        random_state=3857,
        verbose=False,
    ),
    "MLP Grande (256, 128, 64)": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        n_iter_no_change=10,
        early_stopping=True,
        random_state=42,
        verbose=True,
    ),
    "MLP Cilindro (128, 64, 32)": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        n_iter_no_change=10,
        early_stopping=True,
        random_state=42,
        verbose=True,
    ),
    "MLP Diamante (64, 128, 64)": MLPClassifier(
        hidden_layer_sizes=(64, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        n_iter_no_change=10,
        early_stopping=True,
        random_state=42,
        verbose=True,
    ),
    "MLP Leggera (64, 64)": MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        n_iter_no_change=10,
        early_stopping=True,
        random_state=42,
        verbose=True,
    ),
    "MLP Piatta (128,)": MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1000,
        n_iter_no_change=10,
        early_stopping=True,
        random_state=42,
        verbose=True,
    ),
}

TOP4_CLASSIFIERS: Dict[str, Union[ClassifierMixin, LGBMClassifier]] = {
    "KNN": KNeighborsClassifier(n_jobs=-1, n_neighbors=15, weights="distance"),
    "Random Forest": RandomForestClassifier(
        random_state=42, n_estimators=500, max_features="sqrt", n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        verbose=-1,
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=31,
        n_jobs=-1,
    ),
    "MLP Diamante (64, 128, 64)": MLPClassifier(
        hidden_layer_sizes=(64, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=2000,
        n_iter_no_change=10,
        early_stopping=False,
        random_state=42,
        verbose=False,
    ),
}

# ==========================================
# CONFIGURAZIONI PER CLASSIFICAZIONE GERARCHICA
# ==========================================
STAGE1_CLASSIFIERS: Dict[str, Union[ClassifierMixin, LGBMClassifier]] = {
    "lgbm_estremo": LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=63,
        scale_pos_weight=15.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
}

MODEL_STAGE1_BIN: LGBMClassifier = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=63,
    scale_pos_weight=15.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

MODEL_STAGE2_BIN: ClassifierMixin = MLPClassifier(
    hidden_layer_sizes=(64, 256, 64),
    batch_size=1024,
    alpha=0.01,
    max_iter=2000,
    early_stopping=False,
    learning_rate="adaptive",
    random_state=42,
)

HYDROCARBON_THRESHOLD: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8]

# ==========================================
# ARCHITETTURE E CONFIGURAZIONI MLP PER I TEST
# ==========================================
MLP_ARCHITECTURES: List[Tuple[int, ...]] = [
    # 1. ARCHITETTURE "A IMBUTO"
    (256, 128, 64),
    (128, 64, 32),
    (64, 32, 16),
    (32, 16, 8),
    (256, 128, 64, 32, 16),
    (128, 64, 32, 16),
    (64, 32, 16, 8),
    (128, 64, 32, 16, 8),
    (128, 64),
    (64, 32),
    (32, 16),
    (16, 8),
    # Imbuti "Soft"
    (128, 96, 64, 32, 16),
    (64, 48, 32, 16, 8),
    # Imbuti con Plateau
    (256, 256, 128),
    (128, 128, 64),
    (64, 32, 16, 8, 8),
    (32, 16, 8, 8, 8),
    (32, 16, 8, 8),
    (16, 8, 8, 8, 8),
    (16, 8, 8, 8),
    (16, 8, 8),
    # 2. ARCHITETTURE "A DIAMANTE"
    (128, 256, 128),
    (64, 256, 64),
    (64, 128, 64),
    (32, 64, 32),
    (16, 32, 16),
    (8, 16, 8),
    # Diamanti profondi
    (64, 128, 256, 128, 64),
    (32, 64, 128, 64, 32),
    (16, 32, 64, 32, 16),
    (8, 16, 32, 16, 8),
    # Diamanti con Plateau centrale
    (128, 256, 256, 128),
    (64, 128, 128, 64),
    (32, 64, 64, 32),
    (16, 32, 32, 16),
    (8, 16, 16, 8),
    # 3. ARCHITETTURE "WIDE & SHALLOW"
    (1024,),
    (512,),
    (512, 256),
    # 4. ARCHITETTURE "A COLLO DI BOTTIGLIA"
    (128, 16, 128),
    (128, 16, 64, 128),
    (64, 8, 64),
    # 5. ARCHITETTURE "A CILINDRO"
    (128, 128, 128),
    (128, 128),
    (64, 64),
    (32, 32),
    (16, 16),
    (8, 8),
    # 6. ARCHITETTURE "IN ESPANSIONE"
    (64, 128),
    (32, 64),
    (16, 32),
    (8, 16),
    # Espansioni profonde
    (16, 32, 64, 128, 256),
    (16, 32, 64, 128),
    (16, 32, 128),
    (8, 16, 32, 64, 128),
    (8, 16, 32, 64),
    (8, 16, 32),
    # Espansioni con Plateau iniziale
    (8, 8, 16, 32, 64),
    (8, 8, 8, 16, 32),
    (8, 8, 16, 32),
    (8, 8, 8, 16),
    (8, 8, 8, 8, 16),
    (8, 8, 16),
]

TOP_MLP_ARCHITECTURES: List[Tuple[int, ...]] = [
    (1024,),
    (512, 256),
    (64, 128, 64),
    (256, 128, 64, 32, 16),
]

MLP_CONFIGURATIONS: List[Dict[str, Union[float, str]]] = [
    {"alpha": 0.001, "activation": "relu"},
    {"alpha": 0.01, "activation": "relu"},
    {"alpha": 0.1, "activation": "relu"},
    {"alpha": 0.001, "activation": "tanh"},
    {"alpha": 0.01, "activation": "tanh"},
    {"alpha": 0.1, "activation": "tanh"},
]

TOP_HYDROCARBON_MLP_ARCHITECTURES: List[Tuple[int, ...]] = [
    (64, 32),
    (64, 32, 16, 8),
    (64, 256, 64),
    (512,),
]
