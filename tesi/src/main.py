import warnings
from data_manager import DataManager
from experiment_runner import ExperimentRunner
from evaluation import Evaluator
import config

# Ignoriamo i warning di sklearn per mantenere la console pulita durante i test
warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print(" PIPELINE DI MACHINE LEARNING - RIPRODUZIONE RISULTATI TESI")
    print("=" * 80)

    # =========================================================================
    # PANNELLO DI CONTROLLO (Imposta a True le suite che desideri eseguire)
    # =========================================================================
    # FASE 1: Classificazione Flat Multi-Classe
    ESEGUI_MODELLI_BASE = False  # Test algoritmi classici (KNN, Random Forest, ecc.)
    ESEGUI_ARCHITETTURE_MLP = False  # Riproduce i risultati in 'Risultati/Architetture'
    ESEGUI_CONFIGURAZIONI_MLP = (
        False  # Riproduce i risultati in 'Risultati/Configurazioni'
    )

    # FASE 2: Architettura Gerarchica per Sicurezza
    ESEGUI_STAGE1_BINARIO = False  # Riproduce i risultati del filtro binario (LightGBM)
    ESEGUI_ARCHITETTURE_GERARCHICHE = (
        False  # Riproduce 'Risultati/Architetture_idrocarburi'
    )
    ESEGUI_CONFIGURAZIONI_GERARCHICHE = (
        False  # Riproduce 'Risultati/Configurazioni_idrocarburi'
    )
    ESEGUI_SOGLIE_GERARCHICHE = True  # Riproduce 'Risultati/Soglia_idrocarburi'
    # =========================================================================

    # Configurazioni di base per la riproduzione esatta (10-Fold CV su 25 classi)
    ITERAZIONI = 10
    dict_tutti_i_gas = {"TUTTI_I_GAS": config.COMPONENTS}

    # 1. SETUP DEI MODULI
    print("\n[1] Inizializzazione DataManager e Preprocessing in corso...")
    dm = DataManager()

    # feature=True applica media, std, max e la soglia sul rumore ambientale
    dm.preprocess_data(feature=True)

    runner = ExperimentRunner(dm)
    evaluator = Evaluator()

    print(
        f"\n[2] Moduli pronti. Avvio delle suite configurate (Iterazioni K-Fold: {ITERAZIONI})."
    )

    # =========================================================================
    # ESECUZIONE DELLE SUITE DI TEST
    # =========================================================================

    # --- FASE 1: Modelli Classici Singoli (Baseline) ---
    if ESEGUI_MODELLI_BASE:
        runner.run_classifier_suite(
            suite_name="Modelli Base Flat",
            models_dict=config.TOP4_CLASSIFIERS,  # Oppure config.FAST_CLASSIFIERS
            dict_comp=dict_tutti_i_gas,
            evaluator=evaluator,
            n_iter=ITERAZIONI,
            undersampling=True,
        )

    # --- FASE 1: Ricerca Architetturale MLP ---
    if ESEGUI_ARCHITETTURE_MLP:
        runner.suite_architectures_mlp(
            dict_comp=dict_tutti_i_gas, evaluator=evaluator, n_iter=ITERAZIONI
        )

    # --- FASE 1: Ricerca Configurazioni (Grid Search) MLP ---
    if ESEGUI_CONFIGURAZIONI_MLP:
        runner.suite_configurations_mlp(
            dict_comp=dict_tutti_i_gas, evaluator=evaluator, n_iter=ITERAZIONI
        )

    # --- FASE 2: Validazione Stage 1 Binario ---
    if ESEGUI_STAGE1_BINARIO:
        runner.suite_stage1_binary(
            dict_comp=dict_tutti_i_gas,
            evaluator=evaluator,
            n_iter=ITERAZIONI,
            undersampling=True,
        )

    # --- FASE 2: Ricerca Architetturale Specialista Idrocarburi ---
    if ESEGUI_ARCHITETTURE_GERARCHICHE:
        runner.suite_architectures_hierarchical(
            dict_comp=dict_tutti_i_gas, evaluator=evaluator, n_iter=ITERAZIONI
        )

    # --- FASE 2: Ricerca Configurazioni Specialista Idrocarburi ---
    if ESEGUI_CONFIGURAZIONI_GERARCHICHE:
        runner.suite_configurations_hierarchical(
            dict_comp=dict_tutti_i_gas, evaluator=evaluator, n_iter=ITERAZIONI
        )

    # --- FASE 2: Ottimizzazione Soglia di Rigetto (Cestinamento) ---
    if ESEGUI_SOGLIE_GERARCHICHE:
        runner.suite_thresholds_hierarchical(
            dict_comp=dict_tutti_i_gas, evaluator=evaluator, n_iter=ITERAZIONI
        )

    print("\n" + "=" * 80)
    print(" ESECUZIONE DELLA PIPELINE COMPLETATA")
    print(
        " I file di log testuali e le Matrici di Confusione (.png) sono state aggiornate."
    )
    print(" Controlla la cartella 'Risultati/' per visualizzare i dettagli.")
    print("=" * 80)


if __name__ == "__main__":
    main()
