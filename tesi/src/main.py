import warnings
from data_manager import DataManager
from experiment_runner import ExperimentRunner
from evaluation import Evaluator
import config

# Ignoriamo i warning di sklearn per mantenere la console pulita durante i test
warnings.filterwarnings("ignore")


def main():
    print("=" * 70)
    print(" INIZIALIZZAZIONE PIPELINE DI MACHINE LEARNING")
    print("=" * 70)

    # 1. SETUP DEI MODULI
    print("\n[1] Caricamento e Preprocessing dei dati in corso...")
    dm = DataManager()

    # Applichiamo il preprocessing (Feature=True calcola media, std, max orizzontali)
    dm.preprocess_data(feature=True)

    runner = ExperimentRunner(dm)
    evaluator = Evaluator()

    # 2. DEFINIZIONE DEL SOTTOINSIEME DI TEST (Molto rapido)
    # Includiamo un idrocarburo e un non-idrocarburo per testare la gerarchia
    dict_gas_veloce = {"METHANE": ["METHANE"], "AMMONIA": ["AMMONIA"]}

    # Numero di iterazioni ridotto per fare un test rapido del codice
    ITERAZIONI_TEST = 2

    print(
        f"\n[2] Dati pronti. Avvio delle Suite di Test con {ITERAZIONI_TEST} iterazioni."
    )

    # =========================================================================
    # ESECUZIONE DELLE SUITE DI TEST
    # =========================================================================

    # --- TEST 1: Modelli Classici Singoli (Fast Classifiers) ---
    modelli_base_test = {
        "KNN": config.FAST_CLASSIFIERS["KNN"],
        "Decision Tree": config.FAST_CLASSIFIERS["Decision Tree"],
    }
    runner.run_classifier_suite(
        suite_name="Test Modelli Base",
        models_dict=modelli_base_test,
        dict_comp=dict_gas_veloce,
        evaluator=evaluator,
        n_iter=ITERAZIONI_TEST,
    )

    # --- TEST 2: Solo Stage 1 Binario ---
    runner.suite_stage1_binary(
        dict_comp=dict_gas_veloce, evaluator=evaluator, n_iter=ITERAZIONI_TEST
    )

    # --- TEST 3: Architetture Gerarchiche (Stage 1 + Stage 2) ---
    runner.suite_architectures_hierarchical(
        dict_comp=dict_gas_veloce, evaluator=evaluator, n_iter=ITERAZIONI_TEST
    )

    # --- TEST 4: Configurazioni Gerarchiche ---
    runner.suite_configurations_hierarchical(
        dict_comp=dict_gas_veloce, evaluator=evaluator, n_iter=ITERAZIONI_TEST
    )

    # --- TEST 5: Soglie di Confidenza Gerarchiche (Cestinamento) ---
    runner.suite_thresholds_hierarchical(
        dict_comp=dict_gas_veloce, evaluator=evaluator, n_iter=ITERAZIONI_TEST
    )

    print("\n" + "=" * 70)
    print(" TUTTI I TEST SONO STATI COMPLETATI CON SUCCESSO!")
    print(" Controlla la cartella 'Risultati/' per visualizzare i log e le metriche.")
    print("=" * 70)


if __name__ == "__main__":
    main()
