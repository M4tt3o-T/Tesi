"""
Modulo per l'esecuzione degli esperimenti di Machine Learning.
Contiene il motore centrale per la Cross-Validation e l'orchestrazione
delle "Test Suite". Si interfaccia con DataManager per ottenere i dati
e delega a Evaluator il salvataggio di log e grafici.
"""

import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.base import clone, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import config

# Importazioni usate esclusivamente per il Type Hinting
from data_manager import DataManager
from evaluation import Evaluator


class ExperimentRunner:
    """
    Classe che orchestra ed esegue l'addestramento e la validazione dei modelli.

    Attributes:
        dm (DataManager): Istanza per l'accesso e la preparazione dei dati.
        exclude (Dict[str, List[int]]): Memoria degli esperimenti già usati come test.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """
        Inizializza l'esecutore degli esperimenti.

        Args:
            data_manager (DataManager): L'istanza pre-inizializzata per gestire i dati.
        """
        self.dm: DataManager = data_manager
        self.exclude: Dict[str, List[int]] = {c: [] for c in config.COMPONENTS}

    def reset_exclude(self) -> None:
        """Svuota la memoria degli esperimenti utilizzati
        per permettere nuove iterazioni."""
        self.exclude = {c: [] for c in config.COMPONENTS}

    def _apply_majority_voting(
        self, exp_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applica la logica di Majority Voting (Moda) per aggregare le predizioni
        a livello di esperimento.

        Args:
            exp_ids (np.ndarray): Array degli ID associati ad ogni singola riga.
            y_true (np.ndarray): Array delle etichette reali.
            y_pred (np.ndarray): Array delle etichette predette dal modello.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Etichette reali votate,
                Etichette predette votate).
        """
        df_voti = pd.DataFrame(
            {"ID_Esperimento": exp_ids, "Gas_Reale": y_true, "Predizione": y_pred}
        )
        df_voti["Gas_Votato"] = df_voti.groupby("ID_Esperimento")[
            "Predizione"
        ].transform(lambda x: x.mode()[0])
        return df_voti["Gas_Reale"].values, df_voti["Gas_Votato"].values

    # ==========================================
    # MOTORI CENTRALI DI ADDESTRAMENTO E VALIDAZIONE
    # ==========================================

    def run_cross_validation(
        self,
        model_name: str,
        model_instance: ClassifierMixin,
        dict_comp: Dict[str, List[str]],
        n_iter: int = 10,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """
        Motore centrale generico: addestra, vota e calcola tempistiche su N iterazioni.

        Args:
            model_name (str): Nome descrittivo del modello (usato per i log).
            model_instance (ClassifierMixin): L'istanza del classificatore scikit-learn.
            dict_comp (Dict[str, List[str]]): Dizionario dei gas da includere
                nell'esperimento.
            n_iter (int): Numero di fold/iterazioni per la cross-validation.
            **kwargs: Argomenti aggiuntivi da passare al DataManager
                (es. undersampling).

        Returns:
            Tuple[List[Dict], List[str], float]: Lista dei risultati per fold,
                lista delle classi formattate, tempo totale.
        """
        self.reset_exclude()

        classes: List[str] = sorted(
            [
                c
                for v in dict_comp.values()
                if isinstance(v, list)
                for c in v
                if c != "OTHERS"
            ]
        )
        le = LabelEncoder().fit(classes)

        results: List[Dict[str, Any]] = []
        start_time_total = time.time()

        for seed in range(n_iter):
            print(f"--> Iterazione {seed + 1}/{n_iter} ({model_name})...")

            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(
                dict_comp, self.exclude, random_state=seed, **kwargs
            )

            y_enc = le.transform(y)
            model = make_pipeline(StandardScaler(), clone(model_instance))

            # Addestramento e Predizione
            model.fit(X, y_enc)
            y_pred_enc = model.predict(Xts)
            y_pred = le.inverse_transform(y_pred_enc)

            # Voto e Valutazione
            y_true_voted, y_pred_voted = self._apply_majority_voting(
                exp_ids_ts, yts, y_pred
            )
            acc = accuracy_score(y_true_voted, y_pred_voted)

            results.append(
                {
                    "seed": seed,
                    "y_true": y_true_voted,
                    "y_pred": y_pred_voted,
                    "accuracy": acc,
                }
            )

        tempo_totale = time.time() - start_time_total

        return results, classes, tempo_totale

    def run_hierarchical_classification(
        self,
        model_name: str,
        model_s2: ClassifierMixin,
        dict_comp: Dict[str, List[str]],
        n_iter: int = 10,
        threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """
        Motore modulare specifico per la classificazione gerarchica
        (Stage 1 Binario + Stage 2 Specialista).

        Args:
            model_name (str): Nome descrittivo del modello gerarchico.
            model_s2 (ClassifierMixin): L'istanza del modello specialista (Stage 2).
            dict_comp (Dict[str, List[str]]): Dizionario dei gas dell'esperimento.
            n_iter (int): Numero di fold per la cross-validation.
            threshold (float): Soglia di confidenza sotto la quale
                declassare il campione.

        Returns:
            Tuple[List[Dict], List[str], float]: Lista dei risultati dettagliati,
                lista delle classi, tempo totale.
        """
        self.reset_exclude()

        model_s1 = config.MODEL_STAGE1_BIN
        hydrocarbons = config.HYDROCARBONS
        LABEL_NON_HYDRO = "NON_HYDROCARBON"

        classes: List[str] = [LABEL_NON_HYDRO] + sorted(hydrocarbons)

        all_results: List[Dict[str, Any]] = []
        start_time_total = time.time()

        for seed in range(n_iter):
            print(f"--> Iterazione {seed + 1}/{n_iter} ({model_name} - Gerarchico)...")
            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(
                dict_comp, self.exclude, random_state=seed
            )

            # --- STAGE 1 (Idrocarburo vs No) ---
            y_train_bin = np.where(np.isin(y, hydrocarbons), 1, 0)
            pipe_s1 = make_pipeline(StandardScaler(), clone(model_s1))
            pipe_s1.fit(X, y_train_bin)

            y_pred_bin_raw = pipe_s1.predict(Xts)
            _, y_pred_bin_voted = self._apply_majority_voting(
                exp_ids_ts, yts, y_pred_bin_raw
            )

            # --- STAGE 2 (Specialista con Soglia/Cestino) ---
            y_pred_final = np.full(
                y_pred_bin_voted.shape, LABEL_NON_HYDRO, dtype=object
            )
            mask_hydro = y_pred_bin_voted == 1

            if np.any(mask_hydro):
                mask_train_hydro = np.isin(y, hydrocarbons)
                le_s2 = LabelEncoder().fit(y[mask_train_hydro])
                pipe_s2 = make_pipeline(StandardScaler(), clone(model_s2))
                pipe_s2.fit(X[mask_train_hydro], le_s2.transform(y[mask_train_hydro]))

                probs = pipe_s2.predict_proba(Xts[mask_hydro])
                max_probs = np.max(probs, axis=1)
                preds_s2_str = le_s2.inverse_transform(np.argmax(probs, axis=1))

                preds_s2_str[max_probs < threshold] = LABEL_NON_HYDRO

                exp_ids_hydro = exp_ids_ts[mask_hydro]
                _, preds_s2_voted = self._apply_majority_voting(
                    exp_ids_hydro, yts[mask_hydro], preds_s2_str
                )

                y_pred_final[mask_hydro] = preds_s2_voted

            # --- VALUTAZIONE METRICHE ISOLATE ---
            y_true_eval = np.where(np.isin(yts, hydrocarbons), yts, LABEL_NON_HYDRO)
            y_true_binario = np.where(np.isin(yts, hydrocarbons), 1, 0)

            acc_globale = accuracy_score(y_true_eval, y_pred_final)
            acc_s1 = accuracy_score(y_true_binario, y_pred_bin_voted)

            mask_veri_idro_promossi = (y_true_binario == 1) & (y_pred_bin_voted == 1)
            acc_s2 = (
                accuracy_score(
                    yts[mask_veri_idro_promossi], y_pred_final[mask_veri_idro_promossi]
                )
                if np.any(mask_veri_idro_promossi)
                else 0.0
            )

            all_results.append(
                {
                    "seed": seed,
                    "y_true": y_true_eval,
                    "y_pred": y_pred_final,
                    "accuracy": acc_globale,
                    "accuracy_s1": acc_s1,
                    "accuracy_s2": acc_s2,
                }
            )

        tempo_totale = time.time() - start_time_total

        return all_results, classes, tempo_totale

    def run_stage1_binary(
        self,
        model_name: str,
        model_instance: ClassifierMixin,
        dict_comp: Dict[str, List[str]],
        n_iter: int = 10,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """
        Motore centrale isolato per la validazione esclusiva
        dello Stage 1 (Classificatore Binario).

        Args:
            model_name (str): Nome del modello.
            model_instance (ClassifierMixin): L'istanza del modello da testare.
            dict_comp (Dict): Dizionario dei gas.
            n_iter (int): Iterazioni della CV.

        Returns:
            Tuple[List[Dict], List[str], float]: Risultati della matrice 2x2,
                classi ("HYDROCARBON", "NON_HYDROCARBON"), tempo totale.
        """
        self.reset_exclude()

        hydrocarbons = config.HYDROCARBONS
        LABEL_NON_HYDRO = "NON_HYDROCARBON"
        LABEL_HYDRO = "HYDROCARBON"

        classes: List[str] = [LABEL_NON_HYDRO, LABEL_HYDRO]
        results: List[Dict[str, Any]] = []
        start_time_total = time.time()

        for seed in range(n_iter):
            print(
                f"--> Iterazione {seed + 1}/{n_iter}"
                f"({model_name} - Stage 1 Binario)..."
            )

            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(
                dict_comp, self.exclude, random_state=seed, **kwargs
            )

            y_train_bin_num = np.where(np.isin(y, hydrocarbons), 1, 0)
            pipe = make_pipeline(StandardScaler(), clone(model_instance))
            pipe.fit(X, y_train_bin_num)

            y_pred_bin_num = pipe.predict(Xts)

            # Conversione per l'Evaluator
            y_true_str = np.where(
                np.isin(yts, hydrocarbons), LABEL_HYDRO, LABEL_NON_HYDRO
            )
            y_pred_str = np.where(y_pred_bin_num == 1, LABEL_HYDRO, LABEL_NON_HYDRO)

            y_true_voted, y_pred_voted = self._apply_majority_voting(
                exp_ids_ts, y_true_str, y_pred_str
            )
            acc = accuracy_score(y_true_voted, y_pred_voted)

            results.append(
                {
                    "seed": seed,
                    "y_true": y_true_voted,
                    "y_pred": y_pred_voted,
                    "accuracy": acc,
                }
            )

        tempo_totale = time.time() - start_time_total
        return results, classes, tempo_totale

    # ==========================================
    # SUITES (ORCHESTRATORI DI ESPERIMENTI)
    # ==========================================

    def run_classifier_suite(
        self,
        suite_name: str,
        models_dict: Dict[str, ClassifierMixin],
        dict_comp: Dict[str, List[str]],
        evaluator: Evaluator,
        n_iter: int = 10,
        undersampling: bool = True,
    ) -> None:
        """
        Orchestratore universale per eseguire test su una collezione arbitraria
            di modelli standard.

        Si occupa di automatizzare l'intero processo: cicla sui modelli forniti,
            lancia il motore
        di cross-validation per ognuno, delega il salvataggio dei report di testo
            e la generazione
        delle matrici di confusione, e infine estrae la Top 5.

        Args:
            suite_name (str): Nome identificativo della suite (usato per
                creare cartelle e titoli).
            models_dict (Dict[str, ClassifierMixin]):
                Dizionario {nome_modello: istanza_modello}.
            dict_comp (Dict[str, List[str]]): Dizionario dei gas da includere
                nell'esperimento.
            evaluator (Evaluator): L'istanza adibita alla reportistica
                visiva e testuale.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
            undersampling (bool, optional): Se True, bilancia le classi in
                addestramento. Default a True.
        """
        totale_modelli = len(models_dict)
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale_modelli} modelli) ===")

        safe_name = suite_name.replace(" ", "_")
        cartella_output = f"Risultati/{safe_name}"
        nome_file_log = f"{cartella_output}/Risultati_{safe_name}.txt"

        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for nome_modello, modello in models_dict.items():
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")

            risultati, classi, tempo_tot = self.run_cross_validation(
                nome_modello,
                modello,
                dict_comp,
                n_iter=n_iter,
                undersampling=undersampling,
            )

            evaluator.log_suite_result(
                nome_file_log, nome_modello, risultati, tempo_tot
            )

            nome_file_png = f"{cartella_output}/CM_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(
                risultati, classi, nome_modello, nome_file_png
            )
            contatore += 1

        evaluator.generate_top5(
            nome_file_log, f"{cartella_output}/top5_{safe_name}.txt"
        )
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_stage1_binary(
        self,
        dict_comp: Dict[str, List[str]],
        evaluator: Evaluator,
        n_iter: int = 10,
        undersampling: bool = True,
    ) -> None:
        """
        Orchestratore per il test isolato dei classificatori Stage 1 (Binari).

        Verifica esclusivamente la capacità dei modelli definiti in
        `config.STAGE1_CLASSIFIERS` di separare gli Idrocarburi dai Non-Idrocarburi,
        generando matrici di confusione 2x2.

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas da includere.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
            undersampling (bool, optional): Se True, bilancia le classi
                in addestramento. Default a True.
        """
        totale_modelli = len(config.STAGE1_CLASSIFIERS)
        suite_name = "Stage 1 Binario"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale_modelli} modelli) ===")

        safe_name = "Stage1_Binario"
        cartella_output = f"Risultati/{safe_name}"
        nome_file_log = f"{cartella_output}/Risultati_{safe_name}.txt"

        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for nome_modello, modello in config.STAGE1_CLASSIFIERS.items():
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")

            risultati, classi, tempo_tot = self.run_stage1_binary(
                nome_modello,
                modello,
                dict_comp,
                n_iter=n_iter,
                undersampling=undersampling,
            )

            evaluator.log_suite_result(
                nome_file_log, nome_modello, risultati, tempo_tot
            )

            nome_file_png = (
                f"{cartella_output}/CM_Stage1_{nome_modello.replace(' ', '_')}.png"
            )
            evaluator.save_confusion_matrix(
                risultati, classi, nome_modello, nome_file_png
            )
            contatore += 1

        evaluator.generate_top5(
            nome_file_log, f"{cartella_output}/top5_{safe_name}.txt"
        )
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_architectures_mlp(
        self, dict_comp: Dict[str, List[str]], evaluator: Evaluator, n_iter: int = 10
    ) -> None:
        """
        Orchestratore per il collaudo massivo delle architetture MLP standard.

        Fissa tutti gli iperparametri della rete (learning rate, batch size, ecc.)
        e varia esclusivamente la forma della rete leggendo le tuple
        da `config.MLP_ARCHITECTURES`.

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas da analizzare.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
        """
        config_base = {
            "batch_size": 1024,
            "alpha": 0.01,
            "learning_rate_init": 0.001,
            "max_iter": 2000,
            "early_stopping": False,
            "random_state": 42,
        }
        totale_modelli = len(config.MLP_ARCHITECTURES)
        print(f"=== INIZIO TEST ARCHITETTURE ({totale_modelli} reti) ===")

        cartella_output = "Risultati/Architetture"
        nome_file_log = f"{cartella_output}/Risultati_architetture_mlp.txt"
        evaluator.initialize_log_file(nome_file_log, "Report Test Architetture MLP")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            nome_modello = f"MLP {arch}"
            modello = MLPClassifier(hidden_layer_sizes=arch, **config_base)

            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")
            risultati, classi, tempo_tot = self.run_cross_validation(
                nome_modello, modello, dict_comp, n_iter=n_iter, undersampling=True
            )

            evaluator.log_suite_result(
                nome_file_log, nome_modello, risultati, tempo_tot
            )

            # Salva matrice di confusione
            nome_file_png = f"{cartella_output}/CM_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(
                risultati, classi, nome_modello, nome_file_png
            )
            contatore += 1

        evaluator.generate_top5(
            nome_file_log, f"{cartella_output}/top5_architetture.txt"
        )
        print("=== TEST ARCHITETTURE COMPLETATI ===")

    def suite_configurations_mlp(
        self, dict_comp: Dict[str, List[str]], evaluator: Evaluator, n_iter: int = 10
    ) -> None:
        """
        Orchestratore per la Grid Search tra Architetture e Configurazioni MLP standard.

        Testa ogni combinazione possibile incrociando `config.MLP_ARCHITECTURES`
        con i parametri aggiuntivi (come funzioni di attivazione) definiti
        in `config.MLP_CONFIGURATIONS`.

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas da analizzare.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
        """
        totale_modelli = len(config.MLP_ARCHITECTURES) * len(config.MLP_CONFIGURATIONS)
        print(f"=== INIZIO TEST CONFIGURAZIONI ({totale_modelli} reti) ===")

        cartella_output = "Risultati/Configurazioni"
        nome_file_log = f"{cartella_output}/Risultati_configurazioni_mlp.txt"
        evaluator.initialize_log_file(nome_file_log, "Report Test Configurazioni MLP")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            for conf in config.MLP_CONFIGURATIONS:
                nome_modello = (
                    f"MLP {arch} | A:{conf['alpha']} | AF:{conf['activation']}"
                )
                nome_modello_png = (
                    f"MLP {arch} A={conf['alpha']} AF={conf['activation']}"
                )

                modello = MLPClassifier(
                    hidden_layer_sizes=arch,
                    batch_size=1024,
                    max_iter=2000,
                    early_stopping=False,
                    learning_rate="adaptive",
                    random_state=42,
                    **conf,
                )
                print(
                    f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ..."
                )

                risultati, classi, tempo_tot = self.run_cross_validation(
                    nome_modello_png,
                    modello,
                    dict_comp,
                    n_iter=n_iter,
                    undersampling=True,
                )

                evaluator.log_suite_result(
                    nome_file_log, nome_modello, risultati, tempo_tot
                )

                nome_file_png = (
                    f"{cartella_output}/CM_{nome_modello_png.replace(' ', '_')}.png"
                )
                evaluator.save_confusion_matrix(
                    risultati, classi, nome_modello_png, nome_file_png
                )
                contatore += 1

        evaluator.generate_top5(
            nome_file_log, f"{cartella_output}/top5_configurazioni.txt"
        )
        print("=== TEST CONFIGURAZIONI COMPLETATI ===")

    def suite_architectures_hierarchical(
        self, dict_comp: Dict[str, List[str]], evaluator: Evaluator, n_iter: int = 10
    ) -> None:
        """
        Orchestratore per valutare le architetture MLP nel ruolo
        di Specialista (Stage 2).

        Mantiene fisso il modello dello Stage 1 e testa le architetture lette da
        `config.MLP_ARCHITECTURES` esclusivamente sulla loro capacità di discriminare
        i diversi Idrocarburi all'interno del flusso gerarchico.

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
        """
        totale = len(config.MLP_ARCHITECTURES)
        suite_name = "Architetture Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} reti) ===")

        cartella_output = "Risultati/Architetture_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_architetture_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            nome_modello = f"Gerarchico MLP {arch}"

            model_s2 = MLPClassifier(
                hidden_layer_sizes=arch,
                solver="adam",
                batch_size=1024,
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=2000,
                random_state=42,
            )

            print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
            risultati, classi, tempo_tot = self.run_hierarchical_classification(
                nome_modello, model_s2, dict_comp, n_iter=n_iter, threshold=0.0
            )

            evaluator.log_hierarchical_result(
                nome_file_log, nome_modello, risultati, tempo_tot
            )

            nome_file_png = (
                f"{cartella_output}/CM_Gerarchica_{nome_modello.replace(' ', '_')}.png"
            )
            evaluator.save_confusion_matrix(
                risultati, classi, nome_modello, nome_file_png
            )
            contatore += 1

        evaluator.generate_top5_hierarchical(
            nome_file_log, f"{cartella_output}/top5_architetture.txt"
        )
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_configurations_hierarchical(
        self, dict_comp: Dict[str, List[str]], evaluator: Evaluator, n_iter: int = 10
    ) -> None:
        """
        Orchestratore per la Grid Search sulle migliori reti per lo Stage 2.

        Incrocia una selezione ristretta di architetture
        (`config.TOP_HYDROCARBON_MLP_ARCHITECTURES`) con le configurazioni
        (`config.MLP_CONFIGURATIONS`) per ottimizzare l'esperto degli idrocarburi.

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
        """
        totale = len(config.TOP_HYDROCARBON_MLP_ARCHITECTURES) * len(
            config.MLP_CONFIGURATIONS
        )
        suite_name = "Configurazioni Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} reti) ===")

        cartella_output = "Risultati/Configurazioni_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_configurazioni_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for arch in config.TOP_HYDROCARBON_MLP_ARCHITECTURES:
            for conf in config.MLP_CONFIGURATIONS:
                nome_modello = (
                    f"Gerarchico MLP {arch} "
                    f"| A:{conf['alpha']} "
                    f"| AF:{conf['activation']}"
                )
                nome_modello_png = (
                    f"Gerarchico MLP {arch} A={conf['alpha']} AF={conf['activation']}"
                )

                model_s2 = MLPClassifier(
                    hidden_layer_sizes=arch,
                    batch_size=1024,
                    max_iter=2000,
                    learning_rate="adaptive",
                    random_state=42,
                    **conf,
                )

                print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
                risultati, classi, tempo_tot = self.run_hierarchical_classification(
                    nome_modello_png, model_s2, dict_comp, n_iter=n_iter, threshold=0.0
                )

                evaluator.log_hierarchical_result(
                    nome_file_log, nome_modello, risultati, tempo_tot
                )

                nome_file_png = f"{cartella_output}/CM_Gerarchica_{nome_modello_png.replace(' ', '_')}.png"  # noqa E504
                evaluator.save_confusion_matrix(
                    risultati, classi, nome_modello_png, nome_file_png
                )
                contatore += 1

        evaluator.generate_top5_hierarchical(
            nome_file_log, f"{cartella_output}/top5_configurazioni.txt"
        )
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_thresholds_hierarchical(
        self, dict_comp: Dict[str, List[str]], evaluator: Evaluator, n_iter: int = 10
    ) -> None:
        """
        Orchestratore per testare l'efficacia del meccanismo di "cestinamento".

        Mantiene fissi i modelli dello Stage 1 e dello Stage 2, e varia
        progressivamente la soglia di confidenza minima (letta da
        `config.HYDROCARBON_THRESHOLD`) sotto la quale lo Stage 2 rifiuta
        la predizione retrocedendo il campione a "NON_HYDROCARBON".

        Args:
            dict_comp (Dict[str, List[str]]): Dizionario dei gas.
            evaluator (Evaluator): L'istanza adibita alla reportistica.
            n_iter (int, optional): Numero di fold/iterazioni. Default a 10.
        """
        totale = len(config.HYDROCARBON_THRESHOLD)
        suite_name = "Soglie Confidenza Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} soglie) ===")

        cartella_output = "Risultati/Soglia_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_soglia_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        model_s2 = config.MODEL_STAGE2_BIN

        contatore = 1
        for soglia in config.HYDROCARBON_THRESHOLD:
            nome_modello = f"Gerarchico Soglia {soglia}"

            print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
            risultati, classi, tempo_tot = self.run_hierarchical_classification(
                nome_modello, model_s2, dict_comp, n_iter=n_iter, threshold=soglia
            )

            evaluator.log_hierarchical_result(
                nome_file_log, nome_modello, risultati, tempo_tot
            )

            nome_file_png = (
                f"{cartella_output}/CM_Gerarchica_{nome_modello.replace(' ', '_')}.png"
            )
            evaluator.save_confusion_matrix(
                risultati, classi, nome_modello, nome_file_png
            )
            contatore += 1

        evaluator.generate_top5_hierarchical(
            nome_file_log, f"{cartella_output}/top5_soglie.txt"
        )
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")
