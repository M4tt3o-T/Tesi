"""
Modulo per la gestione, il caricamento e il preprocessing dei dati dei sensori.
Si occupa di interfacciarsi con i file fisici (CSV o Pickle), applicare le soglie
di rumore, estrarre feature aggiuntive e generare i set di addestramento e test.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from typing import Dict, List, Tuple

# Importiamo le configurazioni globali
import config


class DataManager:
    """
    Gestore centrale del dataset.

    Legge i dati grezzi dai file, li memorizza in strutture dati efficienti,
    applica filtri di base (es. rimozione rumore) e prepara le matrici (X, y)
    pronte per essere usate dai modelli di Machine Learning.

    Attributes:
        components (List[str]): Lista dei gas da analizzare.
        filenamepickle (str): Percorso del file di cache globale.
        traindir (str): Directory contenente i CSV originali.
        DATA (Dict[Tuple[str, int], Tuple[pd.DataFrame, str]]): Dizionario
            principale grezzo.
        experiments (Dict[str, List[int]]): Mappa ogni gas ai suoi ID esperimento.
        processed_data (Dict[Tuple[str, int], np.ndarray]): Dizionario dei tensori
            post-preprocessing.
    """

    def __init__(self) -> None:
        self.components: List[str] = config.COMPONENTS
        self.filenamepickle: str = config.CACHE_FILE
        self.traindir: str = config.TRAIN_DIR

        # Strutture dati
        self.DATA: Dict[Tuple[str, int], Tuple[pd.DataFrame, str]] = {}
        self.experiments: Dict[str, List[int]] = {}
        self.processed_data: Dict[Tuple[str, int], np.ndarray] = {}

        # Caricamento iniziale dei dati
        if os.path.isfile(self.filenamepickle):
            self.extract_data(export=False)
        else:
            self.extract_data(export=True)

        # Raggruppamento degli ID esperimento per ogni componente
        for component in self.components:
            expr = [b for (a, b) in self.DATA.keys() if a == component]
            self.experiments[component] = expr

    def extract_data(self, export: bool = True) -> None:
        """
        Carica i dati dei sensori in memoria.

        Args:
            export (bool): Se True, legge i CSV e crea il file .npy di cache.
                           Se False, carica direttamente i dati dal file .npy.
        """
        self.DATA = {}
        if export:
            for component in self.components:
                print(f"{component} ", end="")
                traindata = os.path.join(self.traindir, component)

                if os.path.isdir(traindata):
                    expid = 1
                    for file in os.listdir(traindata):
                        if file.endswith(".csv"):
                            print(f"exp{expid} ", end="")
                            file_path = os.path.join(traindata, file)
                            # Memorizza una tupla (DataFrame, nome_file)
                            self.DATA[(component, expid)] = (
                                pd.read_csv(file_path, sep=";"),
                                file,
                            )
                            expid += 1
                print()

            # Salva la cache su disco
            with open(self.filenamepickle, "wb") as filepickle:
                pickle.dump(self.DATA, filepickle)
        else:
            # Carica dalla cache
            with open(self.filenamepickle, "rb") as file:
                self.DATA = pickle.load(file)

    def preprocess_data(
        self,
        feature: bool = True,
        soglia: float = config.SOGLIA,
        sensors: np.ndarray = config.SENSORS,
    ) -> int:
        """
        Applica la soglia per il rumore e calcola feature statistiche orizzontali.

        Args:
            feature (bool): Se True, aggiunge media, deviazione standard
                e massimo di riga.
            soglia (float): Valore di threshold per escludere il segnale
                dell'aria/rumore.
            sensors (np.ndarray): Array degli indici dei sensori da utilizzare.

        Returns:
            int: Il numero finale di feature prodotte (es. 16 senza feature,
                19 con feature).
        """
        self.processed_data = {}

        for (c, e), (df, filename) in self.DATA.items():
            t_raw = df.to_numpy()[:, sensors].astype(np.float64)
            t_raw[pd.isna(t_raw)] = 0.0

            if feature:
                row_mean = np.mean(t_raw, axis=1).reshape(-1, 1)
                row_std = np.std(t_raw, axis=1).reshape(-1, 1)
                row_max = np.max(t_raw, axis=1).reshape(-1, 1)
                t_matrix = np.hstack((t_raw, row_mean, row_std, row_max)).astype(
                    np.float64
                )
            else:
                t_matrix = t_raw

            # Applica la maschera di soglia basata sui dati grezzi
            aa_abs = np.abs(t_raw).sum(axis=1)
            if np.max(aa_abs) > np.min(aa_abs):
                ind = aa_abs > np.min(aa_abs) + soglia * (
                    np.max(aa_abs) - np.min(aa_abs)
                )
                t_matrix = t_matrix[ind, :]

            self.processed_data[(c, e)] = t_matrix

        return t_matrix.shape[1] if bool(self.processed_data) else len(sensors)

    def split_experiments(
        self, c: str, nmin: int = config.NMIN
    ) -> Dict[str, List[int]]:
        """
        Divide gli esperimenti di un gas in tre categorie.

        Args:
            c (str): Nome del componente chimico.
            nmin (int): Numero minimo di righe affinché un esperimento sia valido.

        Returns:
            Dict[str, List[int]]: Dizionario con le chiavi 'NUOVI', 'VECCHI', 'PICCOLI'.
        """
        experiments = {"NUOVI": [], "VECCHI": [], "PICCOLI": []}
        if c not in self.components:
            raise ValueError(f"Component '{c}' not found.")

        for e in self.experiments[c]:
            _, filename = self.DATA[(c, e)]
            t_filtered = self.processed_data.get((c, e), np.array([]))

            if t_filtered.shape[0] <= nmin:
                experiments["PICCOLI"].append(e)
            else:
                if "slow" in filename.lower() or "checkair" in filename.lower():
                    experiments["NUOVI"].append(e)
                else:
                    experiments["VECCHI"].append(e)
        return experiments

    def define_train_test_exp(
        self, c: str, exclude: List[int], tr: str = "SLOW", ts: str = "ALL"
    ) -> Tuple[List[int], List[int]]:
        """
        Assegna gli ID degli esperimenti al Train o al Test set in base alla strategia.

        Args:
            c (str): Nome del componente chimico.
            exclude (List[int]): Lista di ID esperimenti già utilizzati (da ignorare
                nel Test).
            tr (str): Strategia per il Training set ('SLOW', 'STAT', 'ALL', 'ANY').
            ts (str): Strategia per il Test set ('SLOW', 'STAT', 'ALL', 'ANY').

        Returns:
            Tuple[List[int], List[int]]: (Lista esperimenti Train, Lista
                esperimenti Test).
        """
        experiments = self.split_experiments(c)
        list_slow_exp = experiments["NUOVI"]
        list_stat_exp = experiments["VECCHI"] if c in config.LISTA_AMMESSI_STAT else []

        nslow, nstat = len(list_slow_exp), len(list_stat_exp)
        train: List[int] = []
        test: List[int] = []

        if nstat + nslow == 0:
            return train, test

        # Logica di assegnazione Train/Test basata sui parametri
        if tr == "SLOW" and ts == "SLOW":
            lts, nts, ltr, ntr = (
                [list_slow_exp, []],
                [nslow, 0],
                [list_slow_exp, []],
                [nslow, 0],
            )
        elif tr == "SLOW" and ts == "STAT":
            lts, nts, ltr, ntr = (
                [[], list_stat_exp],
                [0, nstat],
                [list_slow_exp, []],
                [nslow, 0],
            )
        elif tr == "SLOW" and ts == "ALL":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, []],
                [nslow, 0],
            )
        elif tr == "SLOW" and ts == "ANY":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, []],
                [nslow, 0],
            )
        elif tr == "STAT" and ts == "SLOW":
            lts, nts, ltr, ntr = (
                [list_slow_exp, []],
                [nslow, 0],
                [[], list_stat_exp],
                [0, nstat],
            )
        elif tr == "STAT" and ts == "STAT":
            lts, nts, ltr, ntr = (
                [[], list_stat_exp],
                [0, nstat],
                [[], list_stat_exp],
                [0, nstat],
            )
        elif tr == "STAT" and ts == "ALL":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [[], list_stat_exp],
                [0, nstat],
            )
        elif tr == "STAT" and ts == "ANY":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [[], list_stat_exp],
                [0, nstat],
            )
        elif tr == "ALL" and ts == "SLOW":
            lts, nts, ltr, ntr = (
                [list_slow_exp, []],
                [nslow, 0],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ALL" and ts == "STAT":
            lts, nts, ltr, ntr = (
                [[], list_stat_exp],
                [0, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ALL" and ts == "ALL":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ALL" and ts == "ANY":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ANY" and ts == "SLOW":
            lts, nts, ltr, ntr = (
                [list_slow_exp, []],
                [nslow, 0],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ANY" and ts == "STAT":
            lts, nts, ltr, ntr = (
                [[], list_stat_exp],
                [0, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ANY" and ts == "ALL":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        elif tr == "ANY" and ts == "ANY":
            lts, nts, ltr, ntr = (
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
                [list_slow_exp, list_stat_exp],
                [nslow, nstat],
            )
        else:
            return train, test  # Paracadute di sicurezza per configurazioni non gestite

        liste: List[int] = []

        # Selezione dei campioni per il Test Set
        for n, l in zip(nts, lts):
            if n > 0:
                count = 0
                while True:
                    i = np.random.choice(np.arange(n))
                    ets = l[i]
                    if ets not in exclude:
                        break
                    count += 1
                    if count >= 100 * n:
                        break
                exclude.append(ets)
                test.append(ets)
                liste.append(ets)

        # Selezione dei campioni rimanenti per il Training Set
        for n, l in zip(ntr, ltr):
            if n > 0:
                for e in l:
                    if e not in liste:
                        train.append(e)

        return train, test

    def generate_trainset(
        self,
        dict_comp: Dict[str, List[str]],
        exclude: Dict[str, List[int]],
        normalized: bool = True,
        balance: bool = True,
        undersampling: bool = True,
        tr: str = "ALL",
        ts: str = "ALL",
        random_state: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assembla le matrici X e y pronte per l'addestramento.

        Args:
            dict_comp: Dizionario dei componenti da utilizzare.
            exclude: Dizionario degli esperimenti da escludere nel test.
            normalized: Se True, normalizza le righe dividendo per la media.
            balance: Se True, fa oversampling della classe minoritaria.
            undersampling: Se True, fa undersampling della classe maggioritaria.
            tr: Strategia per il Training Set.
            ts: Strategia per il Test Set.
            random_state: Seme per la riproducibilità.

        Returns:
            Tuple: X_train, X_test, y_train, y_test, IDs degli esperimenti test.
        """
        np.random.seed(random_state)

        X_list: List[np.ndarray] = []
        Xts_list: List[np.ndarray] = []
        y_list: List[str] = []
        yts_list: List[str] = []
        exp_ids_ts_list: List[str] = []
        cardinality: List[Dict[str, int]] = []

        # Determina il numero di feature dinamicamente dal primo array processato
        if not self.processed_data:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        nsensors = next(iter(self.processed_data.values())).shape[1]

        for comp, comp_list in dict_comp.items():
            if comp == "OTHERS":
                continue

            num, numts = 0, 0
            for c in comp_list:
                ltr, lts = self.define_train_test_exp(c, exclude[c], tr=tr, ts=ts)

                # Popola il Training Set
                for e in ltr:
                    t = self.processed_data[(c, e)]
                    if normalized:
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.0
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))

                    X_list.append(t)
                    y_list.extend([comp] * t.shape[0])
                    num += t.shape[0]

                # Popola il Test Set
                for e in lts:
                    t = self.processed_data[(c, e)]
                    if normalized:
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.0
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))

                    Xts_list.append(t)
                    yts_list.extend([comp] * t.shape[0])
                    exp_ids_ts_list.extend([f"{comp}_{e}"] * t.shape[0])
                    numts += t.shape[0]

            cardinality.append({"comp": comp, "card": num})

        X = np.vstack(X_list) if X_list else np.array([])
        Xts = np.vstack(Xts_list) if Xts_list else np.array([])
        y = np.array(y_list)
        yts = np.array(yts_list)
        exp_ids_ts = np.array(exp_ids_ts_list)

        # Bilanciamento delle classi nel Training Set
        if (undersampling or balance) and len(X) > 0:
            df_card = pd.DataFrame(cardinality)
            target_card = int(df_card["card"].mean())

            X_balanced: List[np.ndarray] = []
            y_balanced: List[np.ndarray] = []

            for _, row in df_card.iterrows():
                comp = str(row["comp"])
                current_card = int(row["card"])
                class_indices = np.where(y == comp)[0]

                if len(class_indices) > 0:
                    if current_card > target_card and undersampling:
                        sampled_indices = np.random.choice(
                            class_indices, size=target_card, replace=False
                        )
                    elif current_card < target_card and balance:
                        sampled_indices = np.random.choice(
                            class_indices, size=target_card, replace=True
                        )
                    else:
                        sampled_indices = class_indices

                    X_balanced.append(X[sampled_indices])
                    y_balanced.append(y[sampled_indices])

            if X_balanced:
                X = np.vstack(X_balanced)
                y = np.concatenate(y_balanced)
                X, y = shuffle(X, y, random_state=random_state)

        return X, Xts, y, yts, exp_ids_ts


if __name__ == "__main__":
    # Semplice test di inizializzazione
    data = DataManager()
    print("DataManager inizializzato con successo!")
