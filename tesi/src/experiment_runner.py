import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import config

class ExperimentRunner:
    def __init__(self, data_manager):
        """
        Args:
            data_manager: Istanza della classe DataManager (già inizializzata)
        """
        self.dm = data_manager
        self.exclude = {c: [] for c in config.COMPONENTS}

    def reset_exclude(self):
        """Pulisce la memoria degli esperimenti usati per il test."""
        self.exclude = {c: [] for c in config.COMPONENTS}

    def _apply_majority_voting(self, exp_ids, y_true, y_pred):
        """Logica di voto per esperimento (Majority Voting)."""
        df_voti = pd.DataFrame({
            'ID_Esperimento': exp_ids,
            'Gas_Reale': y_true,
            'Predizione': y_pred
        })
        # Applica il voto: per ogni ID, prendi la moda delle predizioni
        df_voti['Gas_Votato'] = df_voti.groupby('ID_Esperimento')['Predizione'].transform(lambda x: x.mode()[0])
        return df_voti['Gas_Reale'].values, df_voti['Gas_Votato'].values

    def run_cross_validation(self, model_name, model_instance, dict_comp, n_iter=10, **kwargs):
        """Motore centrale: addestra, vota e calcola le tempistiche."""
        self.reset_exclude()
        
        classes = sorted([c for v in dict_comp.values() if isinstance(v, list) for c in v if c != 'OTHERS'])
        le = LabelEncoder().fit(classes)
        
        results = []
        start_time_total = time.time()  # <-- INIZIO TIMER
        
        for seed in range(n_iter):
            print(f"--> Iterazione {seed + 1}/{n_iter} ({model_name})...")
            
            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(
                dict_comp, self.exclude, random_state=seed, **kwargs
            )
            
            y_enc = le.transform(y)
            model = make_pipeline(StandardScaler(), clone(model_instance))
            
            # FIT & PREDICT
            model.fit(X, y_enc)
            y_pred_enc = model.predict(Xts)
            y_pred = le.inverse_transform(y_pred_enc)
            
            # VOTO
            y_true_voted, y_pred_voted = self._apply_majority_voting(exp_ids_ts, yts, y_pred)
            acc = accuracy_score(y_true_voted, y_pred_voted)
            
            results.append({
                'seed': seed,
                'y_true': y_true_voted,
                'y_pred': y_pred_voted,
                'accuracy': acc
            })
            
        tempo_totale = time.time() - start_time_total  # <-- FINE TIMER
        
        return results, classes, tempo_totale

    def run_hierarchical_classification(self, model_name, model_s2, dict_comp, n_iter=10, threshold=0.0):
        """
        Motore modulare per la classificazione gerarchica (Stage 1 + Stage 2).
        Ora traccia il tempo e calcola le metriche isolate per ogni stage.
        """
        self.reset_exclude()
        
        model_s1 = config.MODEL_STAGE1_BIN
        hydrocarbons = config.HYDROCARBONS
        LABEL_NON_HYDRO = 'NON_HYDROCARBON'
        
        # Le classi finali per la matrice di confusione
        classes = [LABEL_NON_HYDRO] + sorted(hydrocarbons)
        
        all_results = []
        start_time_total = time.time()  # <-- INIZIO TIMER

        for seed in range(n_iter):
            print(f"--> Iterazione {seed + 1}/{n_iter} ({model_name} - Gerarchico)...")
            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(dict_comp, self.exclude, random_state=seed)
            
            # ==========================================
            # STAGE 1 (Binario: Idrocarburo vs No)
            # ==========================================
            y_train_bin = np.where(np.isin(y, hydrocarbons), 1, 0)
            pipe_s1 = make_pipeline(StandardScaler(), clone(model_s1))
            pipe_s1.fit(X, y_train_bin)
            
            y_pred_bin_raw = pipe_s1.predict(Xts)
            _, y_pred_bin_voted = self._apply_majority_voting(exp_ids_ts, yts, y_pred_bin_raw)
            
            # ==========================================
            # STAGE 2 (Specialista Idrocarburi con Cestino)
            # ==========================================
            y_pred_final = np.full(y_pred_bin_voted.shape, LABEL_NON_HYDRO, dtype=object)
            mask_hydro = (y_pred_bin_voted == 1)
            
            if np.any(mask_hydro):
                mask_train_hydro = np.isin(y, hydrocarbons)
                le_s2 = LabelEncoder().fit(y[mask_train_hydro])
                pipe_s2 = make_pipeline(StandardScaler(), clone(model_s2))
                pipe_s2.fit(X[mask_train_hydro], le_s2.transform(y[mask_train_hydro]))
                
                probs = pipe_s2.predict_proba(Xts[mask_hydro])
                max_probs = np.max(probs, axis=1)
                preds_s2_str = le_s2.inverse_transform(np.argmax(probs, axis=1))
                
                preds_s2_str[max_probs < threshold] = LABEL_NON_HYDRO
                
                # Applico il majority voting anche allo Stage 2 per consistenza
                exp_ids_hydro = exp_ids_ts[mask_hydro]
                _, preds_s2_voted = self._apply_majority_voting(exp_ids_hydro, yts[mask_hydro], preds_s2_str)
                
                y_pred_final[mask_hydro] = preds_s2_voted

            # ==========================================
            # VALUTAZIONE METRICHE
            # ==========================================
            y_true_eval = np.where(np.isin(yts, hydrocarbons), yts, LABEL_NON_HYDRO)
            y_true_binario = np.where(np.isin(yts, hydrocarbons), 1, 0)
            
            # Accuratezza Globale
            acc_globale = accuracy_score(y_true_eval, y_pred_final)
            
            # Accuratezza Stage 1
            acc_s1 = accuracy_score(y_true_binario, y_pred_bin_voted)
            
            # Accuratezza Stage 2 (Solo sui veri idrocarburi promossi correttamente)
            mask_veri_idro_promossi = (y_true_binario == 1) & (y_pred_bin_voted == 1)
            acc_s2 = accuracy_score(yts[mask_veri_idro_promossi], y_pred_final[mask_veri_idro_promossi]) if np.any(mask_veri_idro_promossi) else 0.0

            all_results.append({
                'seed': seed,
                'y_true': y_true_eval,
                'y_pred': y_pred_final,
                'accuracy': acc_globale,
                'accuracy_s1': acc_s1,
                'accuracy_s2': acc_s2
            })
            
        tempo_totale = time.time() - start_time_total 
            
        return all_results, classes, tempo_totale
    
    def run_classifier_suite(self, suite_name, models_dict, dict_comp, evaluator, n_iter=10, undersampling=True):
        """
        Motore universale per eseguire una suite di classificatori fornita come dizionario.
        Crea automaticamente le cartelle e i file di log basandosi sul nome della suite.
        """
        totale_modelli = len(models_dict)
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale_modelli} modelli) ===")
        
        # Sostituisce gli spazi con underscore per evitare problemi con i percorsi dei file
        safe_name = suite_name.replace(' ', '_')
        cartella_output = f"Risultati/{safe_name}"
        nome_file_log = f"{cartella_output}/Risultati_{safe_name}.txt"
        
        # Passiamo il percorso all'evaluator (che si occuperà anche di creare la cartella se non esiste)
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for nome_modello, modello in models_dict.items():
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")
            
            # Lanciamo il motore centrale
            risultati, classi, tempo_tot = self.run_cross_validation(
                nome_modello, modello, dict_comp, n_iter=n_iter, undersampling=undersampling
            )
            
            # 1. Log testuale standard
            evaluator.log_suite_result(nome_file_log, nome_modello, risultati, tempo_tot)
            
            # 2. Salvataggio CM (Matrice di confusione completa con tutti i gas)
            nome_file_png = f"{cartella_output}/CM_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(risultati, classi, nome_modello, nome_file_png)
            contatore += 1
            
        # Generiamo la classifica Top 5 a fine ciclo
        evaluator.generate_top5(nome_file_log, f"{cartella_output}/top5_{safe_name}.txt")
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")
    
    def run_stage1_binary(self, model_name, model_instance, dict_comp, n_iter=10, **kwargs):
        """
        Motore centrale per la validazione del solo Stage 1 (Classificatore Binario).
        Addestra su 0/1 ma restituisce stringhe per compatibilità con l'Evaluator.
        """
        self.reset_exclude()
        
        hydrocarbons = config.HYDROCARBONS
        LABEL_NON_HYDRO = 'NON_HYDROCARBON'
        LABEL_HYDRO = 'HYDROCARBON'
        
        # Le classi per l'Evaluator (genereranno una matrice 2x2)
        classes = [LABEL_NON_HYDRO, LABEL_HYDRO]
        
        results = []
        start_time_total = time.time()
        
        for seed in range(n_iter):
            print(f"--> Iterazione {seed + 1}/{n_iter} ({model_name} - Stage 1 Binario)...")
            
            X, Xts, y, yts, exp_ids_ts = self.dm.generate_trainset(
                dict_comp, self.exclude, random_state=seed, **kwargs
            )
            
            # Target binari numerici per l'addestramento (es. LightGBM vuole int)
            y_train_bin_num = np.where(np.isin(y, hydrocarbons), 1, 0)
            
            pipe = make_pipeline(StandardScaler(), clone(model_instance))
            pipe.fit(X, y_train_bin_num)
            
            # Predizione raw (numerica)
            y_pred_bin_num = pipe.predict(Xts)
            
            # Convertiamo y_true e y_pred in stringhe per il calcolo finale
            y_true_str = np.where(np.isin(yts, hydrocarbons), LABEL_HYDRO, LABEL_NON_HYDRO)
            y_pred_str = np.where(y_pred_bin_num == 1, LABEL_HYDRO, LABEL_NON_HYDRO)
            
            # Majority Voting
            y_true_voted, y_pred_voted = self._apply_majority_voting(exp_ids_ts, y_true_str, y_pred_str)
            
            acc = accuracy_score(y_true_voted, y_pred_voted)
            
            results.append({
                'seed': seed,
                'y_true': y_true_voted,
                'y_pred': y_pred_voted,
                'accuracy': acc
            })
            
        tempo_totale = time.time() - start_time_total
        
        return results, classes, tempo_totale

    def suite_stage1_binary(self, dict_comp, evaluator, n_iter=10, undersampling=True):
        """
        Itera sul dizionario STAGE1_CLASSIFIERS nel config e delega
        all'Evaluator il salvataggio dei report e delle matrici 2x2.
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
                nome_modello, modello, dict_comp, n_iter=n_iter, undersampling=undersampling
            )
            
            # 1. Log testuale
            evaluator.log_suite_result(nome_file_log, nome_modello, risultati, tempo_tot)
            
            # 2. Salvataggio CM
            nome_file_png = f"{cartella_output}/CM_Stage1_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(risultati, classi, nome_modello, nome_file_png)
            contatore += 1
            
        evaluator.generate_top5(nome_file_log, f"{cartella_output}/top5_{safe_name}.txt")
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")
    
    def suite_architectures_mlp(self, dict_comp, evaluator, n_iter=10):
        """Esegue i test su tutte le architetture e delega il salvataggio all'Evaluator."""
        
        config_base = {
            'batch_size': 1024, 
            'alpha': 0.01, 
            'learning_rate_init': 0.001, 
            'max_iter': 2000,
            'early_stopping': False,
            'random_state': 42
        }
        totale_modelli = len(config.MLP_ARCHITECTURES)
        
        print(f"=== INIZIO TEST ARCHITETTURE ({totale_modelli} reti) ===")
        
        # Chiediamo all'evaluator di preparare il file (es. scrivere l'intestazione)
        nome_file_log = "Risultati/Architetture/Risultati_architetture_mlp.txt"
        evaluator.initialize_log_file(nome_file_log, "Report Test Architetture MLP")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            nome_modello = f"MLP {arch}"
            
            # Creiamo l'istanza con i parametri base
            modello = MLPClassifier(hidden_layer_sizes=arch, **config_base)
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")
            
            # Lanciamo il motore
            risultati, classi, tempo_tot = self.run_cross_validation(
                nome_modello, modello, dict_comp, n_iter=n_iter, undersampling=True
            )
            
            # L'Evaluator si occupa di calcolare le medie e scrivere sul file
            evaluator.log_suite_result(nome_file_log, nome_modello, risultati, tempo_tot)
            
            contatore += 1

        # Alla fine diciamo all'Evaluator di estrarre la Top 5!
        evaluator.generate_top5(nome_file_log, "Risultati/Architetture/top5_architetture.txt")
        print("=== TEST ARCHITETTURE COMPLETATI ===")
        
    def suite_configurations_mlp(self, dict_comp, evaluator, n_iter=10):
        """Esegue i test su tutte le combinazioni di architetture e configurazioni."""
        
        totale_modelli = len(config.MLP_ARCHITECTURES) * len(config.MLP_CONFIGURATIONS)
        
        print(f"=== INIZIO TEST CONFIGURAZIONI ({totale_modelli} reti) ===")
        
        # Chiediamo all'evaluator di preparare il file
        nome_file_log = "Risultati/Configurazioni/Risultati_configurazioni_mlp.txt"
        evaluator.initialize_log_file(nome_file_log, "Report Test Configurazioni MLP")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            for conf in config.MLP_CONFIGURATIONS:
                # Corretto l'uso della variabile 'conf'
                nome_modello = f"MLP {arch} | A:{conf['alpha']} | AF:{conf['activation']}"
                nome_modello_png = f"MLP {arch} A={conf['alpha']} AF={conf['activation']}"
            
                modello = MLPClassifier(
                    hidden_layer_sizes=arch,
                    batch_size=1024,
                    max_iter=2000, 
                    early_stopping=False,
                    learning_rate='adaptive',
                    random_state=42,
                    **conf
                )
                print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")
            
                risultati, classi, tempo_tot = self.run_cross_validation(
                    nome_modello_png, modello, dict_comp, n_iter=n_iter, undersampling=True
                )
            
                evaluator.log_suite_result(nome_file_log, nome_modello, risultati, tempo_tot)
                
                contatore += 1 # Aggiornamento del contatore

        evaluator.generate_top5(nome_file_log, "Risultati/Configurazioni/top5_configurazioni.txt")
        print("=== TEST CONFIGURAZIONI COMPLETATI ===")
        
    def suite_architectures_hierarchical(self, dict_comp, evaluator, n_iter=10):
        """Testa diverse architetture MLP come specialista Idrocarburi (Stage 2)."""
        totale = len(config.MLP_ARCHITECTURES)
        suite_name = "Architetture Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} reti) ===")

        cartella_output = "Risultati/Architetture_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_architetture_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for arch in config.MLP_ARCHITECTURES:
            nome_modello = f"Gerarchico MLP {arch}"
            
            # Istanza base per i test di architettura
            model_s2 = MLPClassifier(
                hidden_layer_sizes=arch, solver='adam', batch_size=1024,
                alpha=0.01, learning_rate_init=0.001, max_iter=2000, random_state=42
            )

            print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
            risultati, classi, tempo_tot = self.run_hierarchical_classification(
                nome_modello, model_s2, dict_comp, n_iter=n_iter, threshold=0.0
            )

            # 1. Salvataggio del log gerarchico
            evaluator.log_hierarchical_result(nome_file_log, nome_modello, risultati, tempo_tot)
            
            # 2. Salvataggio fisico della Confusion Matrix Gerarchica (.png)
            nome_file_png = f"{cartella_output}/CM_Gerarchica_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(risultati, classi, nome_modello, nome_file_png)
            contatore += 1

        evaluator.generate_top5_hierarchical(nome_file_log, f"{cartella_output}/top5_architetture.txt")
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_configurations_hierarchical(self, dict_comp, evaluator, n_iter=10):
        """Testa le combinazioni di architetture top e configurazioni per lo Stage 2."""
        totale = len(config.TOP_HYDROCARBON_MLP_ARCHITECTURES) * len(config.MLP_CONFIGURATIONS)
        suite_name = "Configurazioni Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} reti) ===")

        cartella_output = "Risultati/Configurazioni_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_configurazioni_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        contatore = 1
        for arch in config.TOP_HYDROCARBON_MLP_ARCHITECTURES:
            for conf in config.MLP_CONFIGURATIONS:
                nome_modello = f"Gerarchico MLP {arch} | A:{conf['alpha']} | AF:{conf['activation']}"
                nome_modello_png = f"Gerarchico MLP {arch} A={conf['alpha']} AF={conf['activation']}"

                model_s2 = MLPClassifier(
                    hidden_layer_sizes=arch, batch_size=1024, max_iter=2000,
                    learning_rate='adaptive', random_state=42, **conf
                )

                print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
                risultati, classi, tempo_tot = self.run_hierarchical_classification(
                    nome_modello_png, model_s2, dict_comp, n_iter=n_iter, threshold=0.0
                )

                # 1. Log testuale gerarchico (usa nome_modello per il testo)
                evaluator.log_hierarchical_result(nome_file_log, nome_modello, risultati, tempo_tot)
                
                # 2. Salvataggio CM (usa nome_modello_png per file e titolo)
                nome_file_png = f"{cartella_output}/CM_Gerarchica_{nome_modello_png.replace(' ', '_')}.png"
                evaluator.save_confusion_matrix(risultati, classi, nome_modello_png, nome_file_png)
                contatore += 1

        evaluator.generate_top5_hierarchical(nome_file_log, f"{cartella_output}/top5_configurazioni.txt")
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")

    def suite_thresholds_hierarchical(self, dict_comp, evaluator, n_iter=10):
        """Testa l'efficacia di varie soglie di 'cestinamento' sullo Stage 2."""
        totale = len(config.HYDROCARBON_THRESHOLD)
        suite_name = "Soglie Confidenza Gerarchiche"
        print(f"\n=== INIZIO TEST {suite_name.upper()} ({totale} soglie) ===")

        cartella_output = "Risultati/Soglia_idrocarburi"
        nome_file_log = f"{cartella_output}/Risultati_soglia_idrocarburi.txt"
        evaluator.initialize_log_file(nome_file_log, f"Report Test: {suite_name}")

        # In questo caso il modello è fisso ed è definito nel config
        model_s2 = config.MODEL_STAGE2_BIN

        contatore = 1
        for soglia in config.HYDROCARBON_THRESHOLD:
            nome_modello = f"Gerarchico Soglia {soglia}"

            print(f"[{contatore}/{totale}] Addestramento: {nome_modello} ...")
            risultati, classi, tempo_tot = self.run_hierarchical_classification(
                nome_modello, model_s2, dict_comp, n_iter=n_iter, threshold=soglia
            )

            # 1. Log testuale gerarchico
            evaluator.log_hierarchical_result(nome_file_log, nome_modello, risultati, tempo_tot)
            
            # 2. Salvataggio CM
            nome_file_png = f"{cartella_output}/CM_Gerarchica_{nome_modello.replace(' ', '_')}.png"
            evaluator.save_confusion_matrix(risultati, classi, nome_modello, nome_file_png)
            contatore += 1

        evaluator.generate_top5_hierarchical(nome_file_log, f"{cartella_output}/top5_soglie.txt")
        print(f"=== TEST {suite_name.upper()} COMPLETATI ===\n")