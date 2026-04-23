# Analisi e Classificazione Gerarchica di Segnali per Sensori di Gas

Questo repository contiene un'architettura software modulare sviluppata per processare, analizzare e classificare serie storiche di dati provenienti da matrici di sensori chimici. 
Il progetto implementa una pipeline completa di Machine Learning, passando dall'estrazione delle feature fino all'approccio di "Classificazione Gerarchica" (con riconoscimento specifico di idrocarburi e logica di rigetto basata su soglie di confidenza).

## Architettura del Progetto

Il codice è stato refattorizzato seguendo i principi di "Single Responsibility" per garantire modularità e facile manutenzione. I moduli comunicano in modo unidirezionale e sono così strutturati:

1. **`config.py` (Configurazione & Modelli):**
   Agisce come singola fonte di verità per l'intero progetto. Contiene percorsi di sistema, parametri di soglia per la pulizia del segnale, liste dei gas e l'inizializzazione di tutti i modelli (Scikit-Learn, LightGBM, XGBoost) e delle architetture di test.

2. **`data_manager.py` (ETL & Preprocessing):**
   Si occupa dell'accesso ai file `.csv` e della cache binaria `.npy`. Applica filtri sul rumore di fondo, calcola feature statistiche (media, varianza) e genera i tensori normalizzati (`X`, `y`) gestendo lo splitting dinamico tra esperimenti di addestramento e test.

3. **`experiment_runner.py` (Motore di Addestramento):**
   Prende i dati preparati, esegue la K-Fold Cross Validation applicando una logica di *Majority Voting* sui frame dei singoli esperimenti, e implementa la pipeline gerarchica a due stadi.

4. **`evaluation.py` (Reportistica & Presentation Layer):**
   Un modulo dedicato esclusivamente all'estetica dei risultati. Riceve metriche brute e le converte in file di log testuali, estrattori di Top 5 e matrici di confusione ad alta risoluzione generate tramite Matplotlib e Seaborn.

5. **`main.py` (Entry Point):**
   Una plancia di comando snella che avvia l'intera pipeline richiamando le suite di test desiderate.

## Requisiti e Installazione

Per eseguire il codice in locale, è necessario Python 3.8+ (o superiore).
Si consiglia di utilizzare un ambiente virtuale (es. `venv` o `conda`).

1. **Clona il repository:**
   ```bash
   git clone https://github.com/M4tt3o-T/Tesi.git
   cd Tesi
   ```

2. **Installa le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preparazione dei Dati:**
   Al primo avvio, il sistema creerà in automatico un file di cache `.npy` per velocizzare i caricamenti successivi.

## ⚙️ Esecuzione degli Esperimenti

L'intero sistema può essere controllato dal file `main.py`. 
Per avviare l'addestramento e generare i risultati, basta eseguire:

```bash
python main.py
```

I risultati, inclusi i report testuali (`.txt`) e le matrici di confusione (`.png`), verranno salvati automaticamente e catalogati all'interno della cartella `Risultati/`.

---
*Progetto sviluppato come tesi di laurea da Matteo Tesei - 2026*