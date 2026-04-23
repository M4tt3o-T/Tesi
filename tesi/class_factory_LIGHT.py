import copy
import os
import re
import sys
import pickle
from contextlib import redirect_stdout
import warnings
import time

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier,  GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tabulate import tabulate

#Automatizza l'intero flusso di lavoro, dalla selezione dei dati alla valutazione dei modelli
class experiment:
    #Configura l'ambiente per un esperimento di classificazione
    """
    Input:
        SCA: Oggetto che contiene i dati (grezzi o normalizzati)
        sensors: Imposta quali dei 16 sensori utilizzare
        nmin: Lunghezza minima di una misurazione per essere considerata valida
        soglia: Usato per filtrare il rumore
        TR, TS: Strategia per il Training Set e il Test Set
        debug: Se True lavora solo su un sottoinsieme di sostanze per velocizzare il test del codice
    """
    def __init__(self,SCA,sensors=range(16),nmin=4,soglia=0.2,TR='SLOW',TS='SLOW',debug=False,components='ALL',feature=True):
        #Inizializza i dati
        self.SCA = SCA
        self.sensors = sensors
        self.nmin = nmin
        self.TR = TR
        self.TS = TS
        self.soglia = soglia
        self.debug = debug
        
        if not feature:
            self.processed_data = {}
            for (c, e), (df, filename) in self.SCA.DATA.items():
                # Estrazione sensori e gestione NaN
                t = df.to_numpy()[:, self.sensors]
                t[pd.isna(t)] = 0.

                # Applicazione della soglia
                aa_abs = np.abs(t).sum(axis=1)
                # Sicurezza: evitiamo errori se il segnale è completamente piatto
                if np.max(aa_abs) > np.min(aa_abs):
                    ind = (aa_abs > np.min(aa_abs) + self.soglia * (np.max(aa_abs) - np.min(aa_abs)))
                    t = t[ind, :]

                # Salviamo l'array numpy filtrato
                self.processed_data[(c, e)] = t
        else:
            self.processed_data = {}

            for (c, e), (df, filename) in self.SCA.DATA.items():
                # 1. Estraiamo solo le colonne dei sensori richiesti come array numpy
                t_raw = df.to_numpy()[:, self.sensors].astype(np.float64)
                t_raw[pd.isna(t_raw)] = 0.

                # 2. CALCOLO DELLE FEATURE ORIZZONTALI (Riga per Riga)
                # asse 1 significa "calcola lungo le colonne (i 16 sensori) per ogni riga"

                # Media dell'array di sensori in quell'istante
                row_mean = np.mean(t_raw, axis=1).reshape(-1, 1)

                # Deviazione standard (Volatilità spaziale della firma chimica)
                row_std = np.std(t_raw, axis=1).reshape(-1, 1)

                # Valore massimo registrato nell'array in quell'istante
                row_max = np.max(t_raw, axis=1).reshape(-1, 1)

                # 3. Concatenazione: (16 Grezzi + 1 Media + 1 Std + 1 Max) = 19 feature
                t_features = np.hstack((t_raw, row_mean, row_std, row_max)).astype(np.float64)

                # 4. Applicazione della Soglia (calcolata in base alla somma dei valori assoluti)
                aa_abs = np.abs(t_raw).sum(axis=1)

                if np.max(aa_abs) > np.min(aa_abs):
                    ind = (aa_abs > np.min(aa_abs) + self.soglia * (np.max(aa_abs) - np.min(aa_abs)))
                    # Applichiamo la maschera alla matrice completa delle feature
                    t_features = t_features[ind, :]

                # Salviamo l'array NumPy arricchito
                self.processed_data[(c, e)] = t_features
            self.sensors = range(t_features.shape[1])

        #Se debug è True imposta un sottoinsieme di sostanze
        if debug:
            list_components = ['AMMONIA', 'RED_WINE', 'KEROSENE']
        #Altrimenti lavora sulla lista completa
        else:
            if components == "ALL":
                list_components = self.SCA.components
            else:
                list_components = components

        #self.determine_air_confusion()

        #Dizionario che conterrà le sostanze idonee
        self.dict_comp = {}
        #Itera sui componenti
        for c in list_components:
            #Chiama un metodo che divide gli esperimenti in 'NUOVI' e 'VECCHI'
            experiments = self.split_experiments(c)
            #Conta il numero di esperimenti 'NUOVI'
            nslow = len(experiments['NUOVI'])
            #Conta il numero di esperimenti 'VECCHI'
            nstat = len(experiments['VECCHI'])
            #Verifica la compatibilità dei dati in base ai parametri TR e TS
            if (TR == 'SLOW' and TS == 'SLOW' and nslow > 0) or \
               (TR == 'SLOW' and TS == 'STAT' and nslow > 0 and nstat > 0) or \
               (TR == 'SLOW' and TS == 'ALL'  and nslow > 0 and nstat > 0) or \
               (TR == 'SLOW' and TS == 'ANY'  and nslow > 0) or \
               (TR == 'STAT' and TS == 'SLOW' and nslow > 0 and nstat > 0) or \
               (TR == 'STAT' and TS == 'STAT' and nstat > 0) or \
               (TR == 'STAT' and TS == 'ALL'  and nslow > 0 and nstat > 0) or \
               (TR == 'STAT' and TS == 'ANY'  and nstat > 0) or \
               (TR == 'ALL'  and TS == 'SLOW' and nslow > 0 and nstat > 0) or \
               (TR == 'ALL'  and TS == 'STAT' and nslow > 0 and nstat > 0) or \
               (TR == 'ALL'  and TS == 'ALL' and nslow > 0 and nstat > 0) or \
               (TR == 'ALL'  and TS == 'ANY' and nslow > 0 and nstat > 0) or \
               (TR == 'ANY' and TS == 'SLOW' and nslow > 0) or \
               (TR == 'ANY' and TS == 'STAT' and nstat > 0) or \
               (TR == 'ANY' and TS == 'ALL' and nslow > 0 and nstat > 0) or \
               (TR == 'ANY' and TS == 'ANY' and (nslow > 0 or nstat > 0)) \
                    :
                #Se una delle condizioni è soddisfatta, aggiunge la sostanza al dizionario dict_comp
                self.dict_comp[c] = [c]
        #Lista temporanea che raccoglie i componenti che non verranno usati nell'esperimento principale
        others = []
        #Dizionario che tiene traccia degli ID degli esperimenti già usati
        self.exclude = {}
        #Dizionario di metadati per sapere che tipo di dati possiede ogni sostanza
        self.misure = {}
        #Itera i componenti
        for c in list_components:
            #Crea una lista vuota in self.exclude che corrisponde a quel componente
            self.exclude[c] = []
            #Flag per sapere se il componente andrà usato
            trovato = False
            #Itera tutti i componenti validi del dizionario
            for i in self.dict_comp:
                #Se quello che stiamo analizzando è presente
                if c in self.dict_comp[i]:
                    #Il componente andrà usato
                    trovato = True
                    #Inizializza delle flag per sapere che mix di dati possiede
                    vecchie = False
                    nuove = False
                    #Itera ogni esperimento associato al componente corrente
                    for e in self.SCA.experiments[c]:
                        #Recupera il nome originale del file
                        (a, b) = self.SCA.DATA[(c, e)]
                        #Se il nome contiene 'slow' o 'checkair', si tratta di misurazioni nuove
                        if (b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1):
                            nuove = True
                        #Altrimenti contiene misurazioni standard
                        else:
                            vecchie = True
                    #Salva quello che ha trovato nel dizionario self.misure
                    self.misure[i] = {'NUOVE':nuove,'VECCHIE':vecchie}
                    #Interrompe il ciclo e passa ad un altro componente
                    break
            #Se il componente non è stato trovato lo aggiunge alla lista di quelli da non usare
            if not trovato:
                others.append(c)
        #Inserisce la lista others nel dizionario dei componenti alla dicitura 'OTHERS'
        self.dict_comp['OTHERS'] = others

        #Inizializza un dizionario con tutti gli oggetti dei modelli che verranno usati
        self.classifiers = {
            'MLP neural net': MLPClassifier(
                hidden_layer_sizes=(256,128,64),
                activation='relu',
                solver='adam',
                alpha=0.01,
                batch_size=1024,
                learning_rate_init=0.001,
                max_iter=30,
                early_stopping=True,
                random_state=42,
                verbose=True
            ),
            'SVM (gaussian)': SVC(kernel='rbf', random_state=42, verbose=False, max_iter=5000),
            'KNN': KNeighborsClassifier(n_jobs=-1),
            'Random Forest': RandomForestClassifier(random_state=42,n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=100),
            'Ridge Classifier': RidgeClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM (Linear Kernel)': SVC(C=10, kernel='linear', probability=False),
            'Linear SVC': LinearSVC(C=10,dual=False,random_state=42,max_iter=1000),
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant': QuadraticDiscriminantAnalysis(reg_param=0.01),
            'XGBoost': XGBClassifier(eval_metric='mlogloss',random_state=42,n_jobs=-1,tree_method='hist'),
            'LightGBM': LGBMClassifier(verbose=-1,random_state=42, n_jobs=-1)
        }
                   
    #Metodo completo per la visualizzazione dei risultati
    def display_results(self,yts_all, ytmp_all, yprd_all, classes, title='Voting cumulative'):
        #Stampa subito un confronto tra etichette vere e votate
        CR = self.report_on_experiment(yts_all, ytmp_all, yprd_all, title)
        #Crea la CM con valori normalizzati
        CM = confusion_matrix(yts_all, yprd_all, normalize='true')
        #Ottiene il dizionario contenente il report
        CR = classification_report(yts_all, yprd_all, zero_division=0,output_dict=True)
        #Inizializza le variabili dei risultati
        rows = []
        accuracy = 0.
        #Itera per ogni metrica di valutazione in CR e costruisce una lista di liste (rows) che rappresenta le righe della futura tabella LaTeX
        for i in CR.keys():
            #Gestisce il caso di 'accuracy' che ha una struttura diversa nel dizionario
            if i == 'accuracy':
                rows.append([i,0.,0.,float(CR[i]),int(CR['macro avg']['support'])])
                accuracy = float(CR[i])
            else:
                rows.append([i,float(CR[i]['precision']),float(CR[i]['recall']),float(CR[i]['f1-score']),int(CR[i]['support'])])

        #Trasforma la lista rows in una tabella LaTeX
        table = tabulate(rows, headers=['Inquinante', 'precision', 'recall', 'f1-score', 'support'], tablefmt='latex', floatfmt=".2f")
        #Salva la CM e la tabella in un file txt
        #with open('CM.txt','w') as file:
        #    np.savetxt(file,CM)
        #    print(f"\nClassification Report ( majority ):",file=file)
        #    #print(classification_report(yts_all, yprd_all, zero_division=0),file=file)
        #    print(table,file=file)

        #Imposta il grafico per stampare la CM normalizzata
        fig = plt.figure(figsize=(14,14), tight_layout=True)
        ax = fig.add_subplot(111)
        cax = ax.matshow(CM, cmap='Blues')
        fig.colorbar(cax)
        ax.xaxis.tick_bottom()
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xticks(ticks=range(len(classes)),labels=classes, rotation=45,ha='right')
        plt.yticks(ticks=range(len(classes)), labels=classes)
        n,m = CM.shape
        for i in range(n):
            for j in range(n):
                plt.text(i-0.3,j+0.3,'{:4.1f}'.format(CM[j,i]*100.),fontsize=7)
        #Scrive l'accuratezza globale in basso
        plt.text(n - 4, -0.7, '{:s} {:4.1f}%'.format('accuracy', accuracy*100.), fontsize=7)
        
        cartella_output = "Risultati"
        os.makedirs(cartella_output, exist_ok=True)
        nome_file = title.replace(" ", "_").replace("/", "-")
        percorso_file = os.path.join(cartella_output, f"{nome_file}_CM.png")
        
        plt.savefig(percorso_file, dpi=600, bbox_inches="tight")
        plt.close()
        
    #Genera un report di valutazione confrontando i dati reali (yts) con quelli predetti (t_temp, y_pred)
    def report_on_experiment(self,yts, y_temp, y_pred, model_name):
        #Stampa la Confusion Matrix basata su y_pred (risultato finale votato)
        print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(yts, y_pred))
        #Stampa un report testuale confrontando la verità con le predizioni aggregate (y_pred)
        print(f"\nClassification Report for {model_name} ( majority ):\n", classification_report(yts, y_pred, zero_division=0))
        #Stampa lo stesso report ma confrontando la verità con le predizioni originali (y_temp)
        print(f"\nClassification Report for {model_name} (one-by-one):\n", classification_report(yts, y_temp, zero_division=0))
        #Restituisce il report con y_pred come dizionario
        return classification_report(yts, y_pred, zero_division=0, output_dict=True)
        
    #Crea le strutture dati con gli esperimenti selezionati
    def generate_trainset(self,dict_comp,exclude,sensors=range(16),normalized=True,balance=True,undersampling=True,perc=1.0,TR='ALL',TS='ALL',random_state=100):
        #TR and TS can either be: 'SLOW', 'STAT', 'ALL'
        #Imposta il seme del generatore casuale
        np.random.seed(random_state)
        #Recupera il numero di sensori
        nsensors = len(sensors)
        #Inizializza le matrici vuote che verranno riempite nel ciclo
        X_list = []
        Xts_list = []
        y_list = []
        yts_list = []
        exp_ids_ts_list = []
        cardinality = []
        #Itera sulle categorie
        for i in dict_comp.keys():
            #Esclude i componenti in 'OTHERS'
            if not i == 'OTHERS':
                cadd = i
                num = 0
                numts = 0
                #Itera sui componenti
                for c in dict_comp[i]:
                    #Crea le liste di dati di train e test per il componente corrente
                    ltr, lts = self.define_train_test_exp(c,exclude[c],TR=TR,TS=TS)
                    #Itera gli esperimenti nella lista di train
                    for e in ltr:
                        # Recupera direttamente i dati filtrati dal pre-processamento
                        t = self.processed_data[(c, e)]
                        #Se i dati vanno normalizzati divide ogni sensore per la media di riga
                        if normalized:
                            aa_mean = t.sum(axis=1) / nsensors
                            aa_mean[np.abs(aa_mean) < 0.1] = 1.
                            t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))
                        n, m = t.shape
                        num += n
                        #Aggiunge i dati processati a X e le etichette a y
                        X_list.append(t)
                        y_list.extend([cadd] * n)
                    #Fa la stessa cosa con i dati nella lista di test
                    for e in lts:
                        t = self.processed_data[(c, e)]
                        if normalized:
                            aa = t.sum(axis=1)/nsensors
                            aa[np.abs(aa) < 0.1] = 1.
                            t = t / np.tile(np.reshape(aa, (-1, 1)), (1, nsensors))
                        n, m = t.shape
                        numts += n
                        Xts_list.append(t)
                        yts_list.extend([cadd] * n)
                        #Creiamo un identificatore per l'esperimento corrente e lo aggiungiamo n volte alla lista
                        unique_exp_id = f"{cadd}_{e}"
                        exp_ids_ts_list.extend([unique_exp_id] * n)
                #Stampa il numero di dati di train e test
                cardinality.append({'comp':i, 'card':num})
                
        X = np.vstack(X_list)
        Xts = np.vstack(Xts_list)
        y = np.array(y_list)
        yts = np.array(yts_list)
        exp_ids_ts = np.array(exp_ids_ts_list)
        df_card = pd.DataFrame(cardinality).sort_values(by=['card', 'comp'], ascending=[False, True])
        
        #Se si vuole fare undersampling o bilanciamento
        if undersampling or balance:
            # Usiamo un valore fisso che corrisponde alla media (il valore fisso è necessario per i test sui sottoinsiemi)
            target_card = int(df_card.iloc[:, -1].mean())
            
            X_balanced = []
            y_balanced = []
            for index, row in df_card.iterrows():
                comp = row['comp']
                current_card = row['card']
                class_indices = np.where(y == comp)[0]
                
                if len(class_indices) > 0:
                    if current_card > target_card and undersampling:
                        # UNDERSAMPLING: Riduciamo la classe maggioritaria (senza cloni)
                        sampled_indices = np.random.choice(class_indices, size=target_card, replace=False)
                    elif current_card < target_card and balance:
                        # BILANCIAMENTO: Se balance è True duplichiamo fino al target
                        sampled_indices = np.random.choice(class_indices, size=target_card, replace=True)
                    else:
                        # Altrimenti teniamo la classe così com'è
                        sampled_indices = class_indices
                        
                    X_balanced.append(X[sampled_indices])
                    y_balanced.append(y[sampled_indices])
            if len(X_balanced) > 0:
                X = np.vstack(X_balanced)
                y = np.concatenate(y_balanced)
                X, y = shuffle(X, y, random_state=random_state)
                
        #Restituisce le matrici complete
        return X, Xts, y, yts, exp_ids_ts
    
    
   
        np.random.seed(random_state)
        #Recupera il numero di sensori
        nsensors = len(sensors)
        #Inizializza le matrici vuote che verranno riempite nel ciclo
        X_list = []
        Xts_list = []
        Xcv_list = []
        y_list = []
        yts_list = []
        ycv_list = []
        exp_ids_ts_list = []
        cardinality = []
        #Itera sulle categorie
        for i in dict_comp.keys():
            #Esclude i componenti in 'OTHERS'
            if not i == 'OTHERS':
                cadd = i
                num = 0
                numts = 0
                #Itera sui componenti
                for c in dict_comp[i]:
                    
                    experiments = {'NUOVI':[], 'VECCHI':[], 'PICCOLI':[]}
                    if c not in self.SCA.components:
                        raise ValueError(f"Component '{c}' not found in list of known components.")
                    for e in self.SCA.experiments[c]:
                        (a,b) = self.SCA.DATA[(c,e)]
                        n,m = a.shape
                        if n <= self.nmin:
                            experiments['PICCOLI'].append(e)
                        else:
                            if (b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1):
                                experiments['NUOVI'].append(e)
                            else:
                                experiments['VECCHI'].append(e)
                    
                    train = []
                    test = []
                    cv = []
                    lt = [experiments['NUOVI'],]
                    nt = [len(experiments['NUOVI']),0]

                    for n,l in zip(nt,lt):
                        if n > 0:
                            count = 0
                            if n != 1:
                                while True:
                                    i = 0
                                    j = 0
                                    while i == j:
                                        i = np.random.choice(np.arange(n))
                                        j = np.random.choice(np.arange(n))
                                    ets = l[i]
                                    ecv = l[j]
                                    if ets not in exclude[c] and ecv not in exclude[c]:
                                        break
                                    count += 1
                                    if count >= 100*n:
                                        break
                                exclude[c].append(ets)
                                test.append(ets)
                                cv.append(ecv)
                            else:
                                test.append(l[0])
                    for n,l in zip(nt,lt):
                        if n > 0:
                            for e in l:
                                if not e in test and not e in cv:
                                    train.append(e)
                    
                    #Itera gli esperimenti nella lista di train
                    for e in train:
                        (a, b) = self.SCA.DATA[(c, e)]
                        t = a.to_numpy()[:, sensors]
                        t[pd.isna(t)] = 0.
                        #Applica la soglia per scartare l'aria e il rumore di fondo
                        aa_abs = np.abs(t).sum(axis=1)
                        ind = (aa_abs > np.min(aa_abs) + self.soglia * (np.max(aa_abs) - np.min(aa_abs)))
                        t = t[ind, :]
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))
                        n, m = t.shape
                        num += n
                        #Aggiunge i dati processati a X e le etichette a y
                        X_list.append(t)
                        y_list.extend([cadd] * n)
                    #Fa la stessa cosa con i dati nella lista di test
                    for e in test:
                        (a, b) = self.SCA.DATA[(c, e)]
                        t = a.to_numpy()[:, sensors]
                        t[pd.isna(t)] = 0.
                        aa = np.abs(t).sum(axis=1)
                        ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                        t = t[ind, :]
                        aa = t.sum(axis=1)/nsensors
                        aa[np.abs(aa) < 0.1] = 1.
                        t = t / np.tile(np.reshape(aa, (-1, 1)), (1, nsensors))
                        n, m = t.shape
                        numts += n
                        Xts_list.append(t)
                        yts_list.extend([cadd] * n)
                        #Creiamo un identificatore per l'esperimento corrente e lo aggiungiamo n volte alla lista
                        unique_exp_id = f"{cadd}_{e}"
                        exp_ids_ts_list.extend([unique_exp_id] * n)
                    for e in cv:
                        (a, b) = self.SCA.DATA[(c, e)]
                        t = a.to_numpy()[:, sensors]
                        t[pd.isna(t)] = 0.
                        #Applica la soglia per scartare l'aria e il rumore di fondo
                        aa_abs = np.abs(t).sum(axis=1)
                        ind = (aa_abs > np.min(aa_abs) + self.soglia * (np.max(aa_abs) - np.min(aa_abs)))
                        t = t[ind, :]
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))
                        n, m = t.shape
                        num += n
                        #Aggiunge i dati processati a X e le etichette a y
                        Xcv_list.append(t)
                        ycv_list.extend([cadd] * n)
                #Stampa il numero di dati di train e test
                cardinality.append({'comp':i, 'card':num})
                
        X = np.vstack(X_list)
        Xts = np.vstack(Xts_list)
        Xcv = np.vstack(Xcv_list)
        y = np.array(y_list)
        yts = np.array(yts_list)
        ycv = np.array(ycv_list)
        exp_ids_ts = np.array(exp_ids_ts_list)
        df_card = pd.DataFrame(cardinality).sort_values(by='card', ascending=False)
        
        # Usiamo un valore fisso che corrisponde alla media (il valore fisso è necessario per i test sui sottoinsiemi)
        target_card = 8000
        
        X_balanced = []
        y_balanced = []
        for index, row in df_card.iterrows():
            comp = row['comp']
            current_card = row['card']
            class_indices = np.where(y == comp)[0]
            
            if len(class_indices) > 0:
                if current_card > target_card:
                    # UNDERSAMPLING: Riduciamo la classe maggioritaria (senza cloni)
                    sampled_indices = np.random.choice(class_indices, size=target_card, replace=False)
                    X_balanced.append(X[sampled_indices])
                elif current_card < target_card:
                    # BILANCIAMENTO: Se balance è True duplichiamo fino al target
                    sampled_indices = np.random.choice(class_indices, size=target_card, replace=True)
                    X_sampled = X[sampled_indices].copy() # Importante: creare una copia
    
                    # Identifichiamo quali indici sono effettivamente dei "cloni" (appaiono più di una volta)
                    _, unique_counts = np.unique(sampled_indices, return_counts=True)
                    if np.any(unique_counts > 1):
                        # Calcoliamo la deviazione standard per ogni sensore su questa specifica classe
                        std_devs = np.std(X[class_indices], axis=0)
                        # Evitiamo deviazioni standard a zero
                        std_devs[std_devs == 0] = 1e-6

                        # Aggiungiamo un rumore gaussiano pari al 5% della deviazione standard del sensore
                        noise = np.random.normal(0, std_devs * 0.05, X_sampled.shape)
                        X_sampled += noise
                        X_balanced.append(X_sampled)
                else:
                    # Altrimenti teniamo la classe così com'è
                    sampled_indices = class_indices
                    X_balanced.append(X[sampled_indices])
                    
                
                y_balanced.append(y[sampled_indices])
        if len(X_balanced) > 0:
            X = np.vstack(X_balanced)
            y = np.concatenate(y_balanced)
            X, y = shuffle(X, y, random_state=random_state)
                
        #Restituisce le matrici complete
        return X, Xts, Xcv, y, yts, ycv, exp_ids_ts
    
    #Prende tutti gli esperimenti per una sostanza e li divide in tre categorie
    def split_experiments(self,c):
        #Inizializza il dizionario finale con tre chiavi vuote
        experiments = {'NUOVI':[], 'VECCHI':[], 'PICCOLI':[]}
        '''dato un componente, restituisce le liste di esperiemti NUOVI e VECCHI separate'''
        #Verifica che la sostanza esista davvero
        if c not in self.SCA.components:
            raise ValueError(f"Component '{c}' not found in list of known components.")

        #Itera gli esperimenti per la sostanza scelta
        for e in self.SCA.experiments[c]:
            #Estrae il dataframe e il nome del file
            (a,b) = self.SCA.DATA[(c,e)]
            #Recupera le dimensioni del dataframe
            t_filtered = self.processed_data[(c, e)]
            n = t_filtered.shape[0]
            #Se il dataframe è troppo piccolo aggiunge l'esperimento corrente a 'PICCOLI'
            if n <= self.nmin:
                experiments['PICCOLI'].append(e)
            #Altrimenti controlla se è 'NUOVO' o 'VECCHIO'
            else:
                if (b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1):
                    experiments['NUOVI'].append(e)
                else:
                    experiments['VECCHI'].append(e)
        #Ritorna il dizionario
        return experiments
    
    #Separa train e test set per un determinato componente (c) e strategia (TR e TS)
    def define_train_test_exp(self,c,exclude,TR='SLOW',TS='ALL'):
        '''crea le liste di esperimenti di train e test secondo del valore di TR e TS
           TR='SLOW' mette nel train solo esperimenti nuovi
           TR='STAT' mette nel train solo esperimenti vecchi
           TR='ALL' mette nel train entrambe le tipologie
           TR='ANY' mette nel train qualunque tipologia
           TS='SLOW' mette nel test un esperimento nuovo
           TS='STAT' mette nel test un esperimento vecchio
           TS='ALL' mette nel test un esperimento nuovo e uno vecchio
           TS='ANY' mette nel test un esperimento nuovo e uno vecchio se presenti'''
        # ['ACETIC_ACID', 'AIR', 'ACETONE', 'AMMONIUM_CHLORIDE', 'AMMONIA', 'CALCIUM_NITRATE', 'DIESEL', 'GASOLINE',
        #  'ETHANOL', 'ISOPROPANOL', 'HYDROGEN_PEROXIDE', 'KEROSENE', 'NITROMETHANE', 'WATER_VAPOR', 'UREA']
        lista_ammessi = ['CALCIUM_NITRATE', 'ISOPROPANOL']
        #Separa esperimenti 'NUOVI' e 'VECCHI' e ne conta il numero
        experiments = self.split_experiments(c)
        list_slow_exp = experiments['NUOVI']
        if c in lista_ammessi:
            list_stat_exp = experiments['VECCHI']
        else:
            list_stat_exp = []
        nstat = len(list_stat_exp)
        nslow = len(list_slow_exp)
        #Inizializza le liste che conterranno gli esperimenti divisi
        train = []
        test = []
        #Controlla che ci siano abbastanza esperimenti per soddisfare le richieste
        if nstat + nslow == 0:
            print('Non ci sono esperimenti!')
            return train,test
        if (TR == 'SLOW' and nslow == 0) or (TS == 'SLOW' and nslow == 0):
            print('Non ci sono esperimenti nuovi')
            return train,test
        if (TR == 'STAT' and nstat == 0) or (TS == 'STAT' and nstat == 0):
            print('Non ci sono esperimenti vecchi')
            return train,test

        #Divide train e test set a seconda delle richieste
        if TR == 'SLOW' and TS == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif TR == 'SLOW' and TS == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif TR == 'SLOW' and TS == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif TR == 'SLOW' and TS == 'ANY':
            lts = [list_slow_exp, list_stat_exp]
            nts = [nslow, nstat]
            ltr = [list_slow_exp, []]
            ntr = [nslow, 0]
        elif TR == 'STAT' and TS == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif TR == 'STAT' and TS == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif TR == 'STAT' and TS == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif TR == 'STAT' and TS == 'ANY':
            lts = [list_slow_exp, list_stat_exp]
            nts = [nslow, nstat]
            ltr = [[], list_stat_exp]
            ntr = [0, nstat]
        elif TR == 'ALL' and TS == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ALL' and TS == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ALL' and TS == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ALL' and TS == 'ANY':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ANY' and TS == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ANY' and TS == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ANY' and TS == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif TR == 'ANY' and TS == 'ANY':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]

        #Lista temporanea che conterrà le scelte per il test set
        liste = []
        #Itera le liste di test
        for n,l in zip(nts,lts):
            #Se ci sono elementi nella lista corrente
            if n > 0:
                count = 0
                #Ciclo che continua fino a che non ha trovato un file accettabile o non ha superato un certo numero di esecuzioni
                while True:
                    #Sceglie a caso un file dalla lista
                    i = np.random.choice(np.arange(n))
                    ets = l[i]
                    #Se non è in exclude allora può essere usato per il test
                    if (ets not in exclude):
                        break
                    #Controllo per evitare cicli infiniti
                    count += 1
                    if count >= 100*n:
                        break
                #Inserisce il file scelto in exclude (per evitare venga scelto di nuovo), in test (la lista finale) e in liste (per il prossimo ciclo)
                exclude.append(ets)
                test.append(ets)
                liste.append(ets)
        #Itera sulle liste di training
        for n,l in zip(ntr,ltr):
            #Se la lista non è vuota
            if n > 0:
                #Itera sugli elementi della lista
                for e in l:
                    #Se il file corrente non è stato scelto per il test, lo aggiunge ai dati di training
                    if not e in liste:
                        train.append(e)

        #Restituisce le liste finali
        return train,test
        
    #Prova i classificatori più veloci
    def prova_fast_classifiers(self,normalized=False,balance=True,perc=1):
        TR = self.TR
        TS = self.TS
        
        out = sys.stdout
        warnings.filterwarnings("ignore")
        
        classifiers = {
            'KNN': KNeighborsClassifier(n_jobs=-1),
            'Random Forest': RandomForestClassifier(random_state=42,n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=100),
            'Ridge Classifier': RidgeClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Linear SVC': LinearSVC(C=10,dual=False,random_state=42,max_iter=1000),
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant': QuadraticDiscriminantAnalysis(reg_param=0.01),
            'XGBoost': XGBClassifier(eval_metric='mlogloss',random_state=42,n_jobs=-1,tree_method='hist'),
            'LightGBM': LGBMClassifier(verbose=-1,random_state=42, n_jobs=-1)
        }
        
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        
            X, Xts, y, yts, _ = self.generate_trainset(self.dict_comp,self.exclude,
                                                        sensors=self.sensors,
                                                        TR=TR,TS=TS,normalized=normalized,
                                                        balance=balance,perc=perc,
                                                        random_state=0)
            
            print(f"[Debug] Train set: {X.shape}, Test set: {Xts.shape}", file=out)

            accuracy = {}
            tempi = {}
            
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            for model_type in classifiers.keys():
                model = make_pipeline(StandardScaler(), classifiers[model_type])

                print(f"Inizio training di {model_type}...", file=out)
                
                start_time = time.time()
                
                model.fit(X, y_enc)
                y_pred = model.predict(Xts)
                
                end_time = time.time()
                
                y_pred = le.inverse_transform(y_pred)
                
                CR = classification_report(yts, y_pred, zero_division=0, output_dict=True)
                
                accuracy[model_type] = CR['accuracy']
                tempi[model_type] = end_time - start_time
            
        modelli_ordinati = sorted(accuracy.items(), key=lambda item: item[1], reverse=True)
        
        print(f"\nRisultati con balance={balance} e normalized={normalized}\n")

        
        for posizione, (model_name, acc) in enumerate(modelli_ordinati, start=1):
            tempo = tempi[model_name]
            print(f"{posizione:2d}. {model_name:<30} | Accuracy: {acc:.4f} | Tempo: {tempo:>8.2f} s")
        print()
            
    #Prova diverse configurazioni di reti neurali
    def prova_mlp(self,normalized=False,balance=True,perc=1):
        TR = self.TR
        TS = self.TS
        
        out = sys.stdout
        warnings.filterwarnings("ignore")
        
        mlps = {
            'MLP Originale (20,)': MLPClassifier(
                hidden_layer_sizes=(20,),
                activation='relu', solver='sgd', alpha=1.e-2,
                max_iter=5000, tol=1.e-3, 
                n_iter_no_change=100,
                random_state=3857, verbose=False),
            
            'MLP Grande (256, 128, 64)': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=1000, n_iter_no_change=10, early_stopping=True, 
                random_state=42, verbose=True
            ),

            'MLP Cilindro (128, 64, 32)': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=1000, n_iter_no_change=10, early_stopping=True, 
                random_state=42, verbose=True
            ),

            'MLP Diamante (64, 128, 64)': MLPClassifier(
                hidden_layer_sizes=(64, 128, 64),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=1000, n_iter_no_change=10, early_stopping=True, 
                random_state=42, verbose=True
            ),

            'MLP Leggera (64, 64)': MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=1000, n_iter_no_change=10, early_stopping=True, 
                random_state=42, verbose=True
            ),

            'MLP Piatta (128,)': MLPClassifier(
                hidden_layer_sizes=(128,),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=1000, n_iter_no_change=10, early_stopping=True, 
                random_state=42, verbose=True
            )
        }
        
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        
            X, Xts, y, yts, _ = self.generate_trainset(self.dict_comp,self.exclude,
                                                        sensors=self.sensors,
                                                        TR=TR,TS=TS,normalized=normalized,
                                                        balance=balance,perc=perc,
                                                        random_state=0)

            print(f"[Debug] Train set: {X.shape}, Test set: {Xts.shape}", file=out)

            accuracy = {}
            tempi = {}
            
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            for mlp in mlps.keys():
                model = make_pipeline(StandardScaler(), mlps[mlp])
                
                print(f"Inizio training di {mlp}...", file=out)
                
                start_time = time.time()
                
                model.fit(X, y_enc)
                y_pred = model.predict(Xts)
                
                end_time = time.time()
                
                y_pred = le.inverse_transform(y_pred)

                CR = classification_report(yts, y_pred, zero_division=0, output_dict=True)
                
                accuracy[mlp] = CR['accuracy']
                tempi[mlp] = end_time - start_time
            
        modelli_ordinati = sorted(accuracy.items(), key=lambda item: item[1], reverse=True)
        
        print(f"\nRisultati con balance={balance} e normalized={normalized}\n")
        
        for posizione, (model_name, acc) in enumerate(modelli_ordinati, start=1):
            tempo = tempi[model_name]
            print(f"{posizione:2d}. {model_name:<30} | Accuracy: {acc:.4f} | Tempo: {tempo:>8.2f} s")
        print()
        
    #Prova i classificatori più lenti
    def prova_slow_classifiers(self,normalized=False,balance=True,perc=1):
        TR = self.TR
        TS = self.TS
        
        out = sys.stdout
        warnings.filterwarnings("ignore")
        
        classifiers = {
            'SVM (gaussian)': SVC(kernel='rbf', random_state=42, verbose=False, max_iter=5000),
            'Gradient Boosting': GradientBoostingClassifier(),
            'SVM (Linear Kernel)': SVC(C=10, kernel='linear', probability=False),
        }
        
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        
            X, Xts, y, yts, _ = self.generate_trainset(self.dict_comp,self.exclude,
                                                        sensors=self.sensors,
                                                        TR=TR,TS=TS,normalized=normalized,
                                                        balance=balance,perc=perc,
                                                        random_state=0)
            
            print(f"[Debug] Train set: {X.shape}, Test set: {Xts.shape}", file=out)

            accuracy = {}
            tempi = {}
            
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            for model_type in classifiers.keys():
                model = make_pipeline(StandardScaler(), classifiers[model_type])

                print(f"Inizio training di {model_type}...", file=out)
                
                start_time = time.time()
                
                model.fit(X, y_enc)
                y_pred = model.predict(Xts)
                
                end_time = time.time()
                y_pred = le.inverse_transform(y_pred)
                CR = classification_report(yts, y_pred, zero_division=0, output_dict=True)
                
                accuracy[model_type] = CR['accuracy']
                tempi[model_type] = end_time - start_time
            
        modelli_ordinati = sorted(accuracy.items(), key=lambda item: item[1], reverse=True)
        
        print(f"\nRisultati con balance={balance} e normalized={normalized}\n")

        for posizione, (model_name, acc) in enumerate(modelli_ordinati, start=1):
            tempo = tempi[model_name]
            
            print(f"{posizione:2d}. {model_name:<30} | Accuracy: {acc:.4f} | Tempo: {tempo:>8.2f} s")
        print()
        
        
    def prova_cross_validation(self,nome_modello,istanza_modello,nome='',n_iterazioni=10,normalized=True,balance=True,undersampling=False,perc=1,dir=''):
        """
        Esegue N iterazioni del modello cambiando lo split (random_state),
        calcola la Confusion Matrix media e la deviazione standard, e salva il grafico.
        """
        
        warnings.filterwarnings("ignore")
        #Pulisce la memoria ad ogni test
        for c in self.exclude:
            self.exclude[c] = []
        
        print("\n" + "="*70)
        print(f"Avvio Validazione a {n_iterazioni} Iterazioni: {nome_modello}")
        print("="*70)
        
        #Recuperiamo la lista GLOBALE delle classi in modo che la matrice sia sempre delle stesse dimensioni
        classes_list = []
        for v in self.dict_comp.values():
            if isinstance(v, list):
                for c in v:
                    if c != 'OTHERS' and c not in classes_list:
                        classes_list.append(c)
        classes = sorted(classes_list)
        
        le = LabelEncoder()
        le.fit(classes)
        
        all_cms = []
        accuracies = []
        acc_no_voting = []
        start_time_total = time.time()
        
        #Ciclo delle Iterazioni
        for seed in range(n_iterazioni):
            print(f"--> Esecuzione Iterazione {seed + 1}/{n_iterazioni} (random_state={seed})...")
            
            #Generiamo il dataset con un random_state diverso ad ogni iterazione
            X, Xts, y, yts, exp_ids_ts = self.generate_trainset(
                self.dict_comp, self.exclude, sensors=self.sensors,
                TR=self.TR, TS=self.TS, normalized=normalized,
                balance=balance, perc=perc, random_state=seed, undersampling=undersampling
            )
            y_enc = le.transform(y)
            model = make_pipeline(StandardScaler(), clone(istanza_modello))
            model.fit(X, y_enc)
            
            y_pred_enc = model.predict(Xts)
            y_pred_singolo = le.inverse_transform(y_pred_enc)
            
            acc_singola = accuracy_score(yts, y_pred_singolo)
            acc_no_voting.append(acc_singola)
            
            #Majority Voting con Pandas
            df_voti = pd.DataFrame({
                'ID_Esperimento': exp_ids_ts,
                'Gas_Reale': yts,
                'Predizione_Singola': y_pred_singolo
            })
            df_voti['Gas_Votato'] = df_voti.groupby('ID_Esperimento')['Predizione_Singola'].transform(lambda x: x.mode()[0])
            
            y_true_votato = df_voti['Gas_Reale'].values
            y_pred_votato = df_voti['Gas_Votato'].values
            
            #Calcolo Accuracy di questa specifica iterazione
            acc = accuracy_score(y_true_votato, y_pred_votato)
            accuracies.append(acc)
            print(f"Accuracy Singola: {(acc_singola*100):.2f}%")
            print(f"Accuracy Votata: {(acc*100):.2f}%")
            
            #Calcolo Confusion Matrix normalizzata e conversione in percentuale
            cm = confusion_matrix(y_true_votato, y_pred_votato, labels=classes, normalize='true')
            all_cms.append(cm * 100)
            
        #Aggregazione dei Risultati
        mean_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        tempo_totale = time.time() - start_time_total
        print(f"\nValidazione completata in {tempo_totale:.2f} secondi!")
        print(f"Accuratezza Media Singola: {(np.mean(acc_no_voting) * 100):.2f}% ± {(np.std(acc_no_voting) * 100):.2f}%")
        print(f"Accuratezza Media Votata: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        #Creiamo la matrice di testo per le annotazioni (mostriamo solo se la media > 0.1%)
        annot = np.empty_like(mean_cm, dtype=object)
        n = len(classes)
        for i in range(n):
            for j in range(n):
                if mean_cm[i, j] >= 0.1:
                    annot[i, j] = f"{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}"
                else:
                    annot[i, j] = ""

        #Impostazione Grafica
        fig, ax = plt.subplots(figsize=(16, 14), tight_layout=True)
        sns.heatmap(mean_cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                    xticklabels=classes, yticklabels=classes, ax=ax, 
                    annot_kws={"size": 8},
                    linewidths=0.5, linecolor="lightgray")
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        #Titolo con l'accuratezza media
        accuracy_string = f"{mean_acc:>5.1f}% ± {std_acc:>4.1f}%"
        titolo_grafico = f"{n_iterazioni}-Fold CV: {nome_modello}\nAccuracy: {accuracy_string}"
        plt.title(titolo_grafico, fontsize=18, fontweight='bold', pad=20)
        plt.ylabel("Etichetta Reale", fontweight='bold', fontsize=12)
        plt.xlabel("Etichetta Predetta", fontweight='bold', fontsize=12)
        
        #Salvataggio
        os.makedirs("Risultati", exist_ok=True)
        nome_file = f"Risultati/{dir}CV_{nome}_{nome_modello.replace(' ', '_')}_CM.png"
        plt.savefig(nome_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Grafico salvato in: {nome_file}\n")
        
        minuti, secondi = divmod(int(tempo_totale), 60)
        ore, minuti = divmod(minuti, 60)
        tempo_string = f"{ore:02d}:{minuti:02d}:{secondi:02d}"
        
        return accuracy_string, tempo_string
          
         
    def prova_top4_cross_validation(self,n_iterazioni,nome='',undersampling=False):
        
        #Definiamo i classificatori migliori
        classifiers = {
            'KNN': KNeighborsClassifier(n_jobs=-1, n_neighbors=15, weights="distance"),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=500, max_features='sqrt', n_jobs=-1),
            'LightGBM': LGBMClassifier(verbose=-1, random_state=42, n_estimators=200, learning_rate=0.05, max_depth=10, num_leaves=31, n_jobs=-1),
            'MLP Diamante (64, 128, 64)': MLPClassifier(
                hidden_layer_sizes=(64, 128, 64),
                activation='relu', solver='adam', alpha=0.01,
                batch_size=1024, learning_rate_init=0.001,
                max_iter=2000, n_iter_no_change=10, early_stopping=False, 
                random_state=42, verbose=False
            )
        }
        
        res = {}
        #Eseguiamo il test con la cross validation per ognuno dei classificatori sopra
        for model_type in classifiers.keys():
            temp = self.prova_cross_validation(nome_modello=model_type, istanza_modello=classifiers[model_type], nome=nome, n_iterazioni=n_iterazioni, undersampling=undersampling)
            res[model_type] = temp
        
        print("Risultati:")
        for model_type in classifiers.keys():
            print(f"{model_type+":":<30} {res[model_type][0]} {res[model_type][1]}")
        print()
    
    
    def prova_strati_mlp(self):
        architetture = [
            # 1. ARCHITETTURE "A IMBUTO"
            (256, 128, 64), (128, 64, 32), (64, 32, 16), (32, 16, 8),
            (256, 128, 64, 32, 16), (128, 64, 32, 16), (64, 32, 16, 8), 
            (128, 64, 32, 16, 8), (128, 64), (64, 32), (32, 16), (16, 8),
            # Imbuti "Soft"
            (128, 96, 64, 32, 16), (64, 48, 32, 16, 8),
            # Imbuti con Plateau
            (256, 256, 128), (128, 128, 64), (64, 32, 16, 8, 8),
            (32, 16, 8, 8, 8), (32, 16, 8, 8), (16, 8, 8, 8, 8), 
            (16, 8, 8, 8), (16, 8, 8),
        
            # 2. ARCHITETTURE "A DIAMANTE"
            (128, 256, 128), (64, 256, 64), (64, 128, 64), (32, 64, 32), 
            (16, 32, 16), (8, 16, 8),
            # Diamanti profondi
            (64, 128, 256, 128, 64), (32, 64, 128, 64, 32), (16, 32, 64, 32, 16), 
            (8, 16, 32, 16, 8),
            # Diamanti con Plateau centrale
            (128, 256, 256, 128), (64, 128, 128, 64), (32, 64, 64, 32), 
            (16, 32, 32, 16), (8, 16, 16, 8),
        
            # 3. ARCHITETTURE "WIDE & SHALLOW"
            (1024,), (512,), (512, 256), 
        
            # 4. ARCHITETTURE "A COLLO DI BOTTIGLIA"
            (128, 16, 128), (128, 16, 64, 128), (64, 8, 64),
        
            # 5. ARCHITETTURE "A CILINDRO"
            (128, 128, 128), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8),
        
            # 6. ARCHITETTURE "IN ESPANSIONE"
            (64, 128), (32, 64), (16, 32), (8, 16),
            # Espansioni profonde
            (16, 32, 64, 128, 256), (16, 32, 64, 128), (16, 32, 128),
            (8, 16, 32, 64, 128), (8, 16, 32, 64), (8, 16, 32),
            # Espansioni con Plateau iniziale
            (8, 8, 16, 32, 64), (8, 8, 8, 16, 32), (8, 8, 16, 32), 
            (8, 8, 8, 16), (8, 8, 8, 8, 16), (8, 8, 16)
        ]      
        
        # I parametri base da mantenere costanti durante il confronto delle architetture
        config_base = {
            'batch_size': 1024, 
            'alpha': 0.01, 
            'learning_rate_init': 0.001, 
            'max_iter': 2000,
            'early_stopping': False,
            'random_state': 42
        }       
        nome_file = "Risultati/Architetture/Risultati_architetture_mlp.txt"
        totale_modelli = len(architetture)

        print("=== INIZIO TEST ARCHITETTURE ===")
        print(f"Totale reti da addestrare: {totale_modelli}")

        with open(nome_file, "w", encoding="utf-8") as file:
            intestazione = f"Report Test Architetture MLP\n"
            file.write(intestazione + "="*130 + "\n")      
        contatore = 1       
        for arch in architetture:
            nome_modello = f"MLP {arch}"

            modello = MLPClassifier(hidden_layer_sizes=arch, **config_base)
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")

            accuratezza_str, tempo_str = self.prova_cross_validation(nome_modello=nome_modello, istanza_modello=modello, n_iterazioni=10, undersampling=True, dir='Architetture/')
            riga_risultato = f"{nome_modello:<30} -> Accuracy: {accuratezza_str:<15} | Tempo: {tempo_str}"
            print(f"   Completato! Accuracy: {accuratezza_str} | Tempo: {tempo_str}\n")
         
            # Salvataggio immediato
            with open(nome_file, "a", encoding="utf-8") as file:
                file.write(riga_risultato + "\n")

            contatore += 1      
        print("=========================================================================================")
        print(f"=== TEST COMPLETATI ===")

        self.genera_top5(nome_file, "Risultati/Architetture/top5_architetture.txt")
        print("Classifica generata nel file: top5_architetture.txt")
    
    
    def prova_configurazioni_mlp(self):
        architetture = [
            (1024),
            (512, 256),
            (64, 128, 64),
            (256, 128, 64, 32, 16)
        ]

        configurazioni = [
            { 'alpha': 0.001, 'activation': 'relu' },
            { 'alpha': 0.01,  'activation': 'relu' },
            { 'alpha': 0.1,   'activation': 'relu' },
            { 'alpha': 0.001, 'activation': 'tanh' },
            { 'alpha': 0.01,  'activation': 'tanh' },
            { 'alpha': 0.1,   'activation': 'tanh' }
        ]

        nome_file = "Risultati/Configurazioni/Risultati_configurazioni_mlp.txt"
        totale_modelli = len(architetture) * len(configurazioni)

        print("=== INIZIO ESPERIMENTI MLP ===")
        print(f"Totale modelli da testare: {totale_modelli}")
        print(f"I risultati verranno salvati in: {nome_file}\n")

        # Gestione log continuo
        with open(nome_file, "w", encoding="utf-8") as file:
            intestazione = f"Report Unico Esperimenti MLP\n"
            intestazione += "="*130 + "\n"
            file.write(intestazione)

        contatore = 1

        # Loop di addestramento
        for arch in architetture:
            for config in configurazioni:
                nome_modello = (f"MLP {arch} | A:{config['alpha']} | AF:{config['activation']}")
                nome_modello_png = (f"MLP {arch} A={config['alpha']} AF={config['activation']}")

                modello = MLPClassifier(
                    hidden_layer_sizes=arch,
                    batch_size=1024,
                    max_iter=2000, 
                    early_stopping=False,
                    learning_rate='adaptive',
                    random_state=42,
                    **config
                )

                print(f"[{contatore}/{totale_modelli}] Addestramento in corso: {nome_modello} ...")

                accuratezza_str, tempo_str = self.prova_cross_validation(nome_modello=nome_modello_png, istanza_modello=modello, n_iterazioni=10, undersampling=True, dir="Configurazioni2/")

                riga_risultato = f"{nome_modello:<50} -> Accuracy: {accuratezza_str:<15} | Tempo: {tempo_str}"
                print(f"Completato! Accuracy: {accuratezza_str} | Tempo: {tempo_str}\n")

                with open(nome_file, "a", encoding="utf-8") as file:
                    file.write(riga_risultato + "\n")

                contatore += 1

        print("=========================================================================================")
        print(f"=== TUTTI GLI ESPERIMENTI SONO CONCLUSI CON SUCCESSO ===")

        # Generazione automatica del file Top 5
        self.genera_top5(nome_file, "Risultati/Configurazioni2/top5_configurazioni.txt")
      
        
    def genera_top5(self, file_log, file_output):
        """Legge il log, estrae le percentuali e salva le prime 5 in un nuovo file."""
        
        if not os.path.exists(file_log):
            print(f"Errore: Il file {file_log} non esiste. Impossibile generare la Top 5.")
            return

        risultati = []
        # La regex cerca "Accuracy: ", poi cattura numeri e punti decimali, seguiti da "%"
        pattern_acc = re.compile(r"Accuracy:\s*([0-9\.]+)%")

        with open(file_log, "r", encoding="utf-8") as f:
            for linea in f:
                match = pattern_acc.search(linea)
                if match:
                    # Estraiamo il valore float per l'ordinamento numerico
                    acc_val = float(match.group(1))
                    risultati.append((acc_val, linea.strip()))

        # Ordina la lista di tuple basandosi sul primo elemento (l'accuratezza) in ordine decrescente
        risultati_ordinati = sorted(risultati, key=lambda x: x[0], reverse=True)
        top5 = risultati_ordinati[:5]

        # Scrive la classifica nel file output
        with open(file_output, "w", encoding="utf-8") as f:
            intestazione = f"=== TOP 5 CONFIGURAZIONI MLP  ===\n"
            f.write(intestazione + "="*130 + "\n")

            for i, (_, riga) in enumerate(top5, 1):
                f.write(f"{i}° Posto | {riga}\n")

        print(f"\nClassifica 'Top 5' generata con successo nel file: {file_output}")
    
    
    def prova_stage1_binario(self, nome='', n_iterazioni=10, normalized=True, balance=True, undersampling=True, perc=1):
        """
        Esegue N iterazioni per testare SOLO il classificatore binario (Stage 1).
        Calcola la Confusion Matrix media 2x2 (HYDROCARBON vs NON_HYDROCARBON).
        """
        
        warnings.filterwarnings("ignore")
        
        nome_modello = 'lgbm_estremo'
        model_stage1 = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=63,
            scale_pos_weight=15.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        print("\n" + "="*70)
        print(f"Avvio Validazione Stage 1 (Binario) a {n_iterazioni} Iterazioni: {nome_modello}")
        print("="*70)
        
        # 1. Definizione delle classi per la valutazione
        hydrocarbons = ['METHANE', 'BUTANE', 'GASOLINE', 'LIGHTER_FLUID', 'DIESEL', 'KEROSENE']
        LABEL_NON_HYDRO = 0
        LABEL_HYDRO = 1
        
        # La matrice sarà rigorosamente 2x2
        classes_eval = [LABEL_NON_HYDRO, LABEL_HYDRO]
        
        all_cms = []
        accuracies = []
        acc_no_voting = []
        start_time_total = time.time()
        
        # Ciclo delle Iterazioni
        for seed in range(n_iterazioni):
            print(f"--> Esecuzione Iterazione {seed + 1}/{n_iterazioni} (random_state={seed})...")
            
            # Generiamo il dataset
            X, Xts, y, yts, exp_ids_ts = self.generate_trainset(
                self.dict_comp, self.exclude, sensors=self.sensors,
                TR=self.TR, TS=self.TS, normalized=normalized,
                balance=balance, perc=perc, random_state=seed, undersampling=undersampling
            )
            
            # ==========================================
            # STAGE 1: Addestramento Modello Binario
            # ==========================================
            # Assegniamo DIRETTAMENTE i numeri: 1 per Idrocarburi, 0 per il resto
            y_train_bin_enc = np.where(np.isin(y, hydrocarbons), 1, 0)
            y_true_binario_ts = np.where(np.isin(yts, hydrocarbons), 1, 0)
            
            pipe_stage1 = make_pipeline(StandardScaler(), clone(model_stage1))
            pipe_stage1.fit(X, y_train_bin_enc)
            
            # ==========================================
            # STAGE 1: PREDIZIONE E VOTO BINARIO
            # ==========================================
            y_pred_bin_enc_singolo = pipe_stage1.predict(Xts)
            
            # Applichiamo il Voting Logic per Esperimento sui numeri
            df_voti_s1 = pd.DataFrame({
                'ID_Esperimento': exp_ids_ts,
                'Pred_Bin_Singola': y_pred_bin_enc_singolo
            })
            
            df_voti_s1['Voto_Binario'] = df_voti_s1.groupby('ID_Esperimento')['Pred_Bin_Singola'].transform(lambda x: x.mode()[0])
            y_pred_bin_votato = df_voti_s1['Voto_Binario'].values
            
            # ==========================================
            # AGGIORNAMENTO METRICHE
            # ==========================================
            acc_singola = accuracy_score(y_true_binario_ts, y_pred_bin_enc_singolo)
            acc_no_voting.append(acc_singola)
            
            acc_votata = accuracy_score(y_true_binario_ts, y_pred_bin_votato)
            accuracies.append(acc_votata)
            
            print(f"    Accuracy Singola: {(acc_singola*100):.2f}% | Votata: {(acc_votata*100):.2f}%")
            
            # Calcolo Confusion Matrix 2x2
            cm = confusion_matrix(y_true_binario_ts, y_pred_bin_votato, labels=classes_eval, normalize='true')
            all_cms.append(cm * 100)
            
        # ==========================================
        # AGGREGAZIONE DEI RISULTATI E STAMPA
        # ==========================================
        mean_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        tempo_totale = time.time() - start_time_total
        print(f"\nValidazione completata in {tempo_totale:.2f} secondi!")
        print(f"Accuratezza Media Singola (Binaria): {(np.mean(acc_no_voting) * 100):.2f}% ± {(np.std(acc_no_voting) * 100):.2f}%")
        print(f"Accuratezza Media Votata (Binaria): {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Annotazioni per la Heatmap
        annot = np.empty_like(mean_cm, dtype=object)
        n = len(classes_eval)
        for i in range(n):
            for j in range(n):
                if mean_cm[i, j] >= 0.1:
                    annot[i, j] = f"{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}"
                else:
                    annot[i, j] = ""

        # Impostazione Grafica (Più piccola, dato che è solo 2x2)
        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        import seaborn as sns
        classes_eval = ['NON_HYDROCARBON', 'HYDROCARBON']
        sns.heatmap(mean_cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                    xticklabels=classes_eval, yticklabels=classes_eval, ax=ax, 
                    annot_kws={"size": 14},
                    linewidths=0.5, linecolor="lightgray")
        
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        accuracy_string = f"{mean_acc:>5.1f}% ± {std_acc:>4.1f}%"
        titolo_grafico = f"{n_iterazioni}-Fold CV: {nome_modello}\nAccuracy Binaria: {accuracy_string}"
        plt.title(titolo_grafico, fontsize=12, fontweight='bold', pad=15)
        plt.ylabel("Etichetta Reale", fontweight='bold', fontsize=11)
        plt.xlabel("Etichetta Predetta", fontweight='bold', fontsize=11)
        
        os.makedirs("Risultati", exist_ok=True)
        nome_file = f"Risultati/CV_Stage1_{nome}_{nome_modello.replace(' ', '_')}_CM.png"
        plt.savefig(nome_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Grafico salvato in: {nome_file}\n")
        
        minuti, secondi = divmod(int(tempo_totale), 60)
        ore, minuti = divmod(minuti, 60)
        tempo_string = f"{ore:02d}:{minuti:02d}:{secondi:02d}"
        
        return accuracy_string, tempo_string
     
    
    def classificatore_idrocarburi(self, model_stage1, model_stage2, nome_modello='Gerarchica', nome='', n_iterazioni=10, soglia_confidenza_s2=0.0,dir=''):
        """
        Esegue N iterazioni della Classificazione Gerarchica.
        Include lo Stage 1 binario (numerico) e lo Stage 2 con logica di "cestinamento" basata sulla confidenza.
        """
        warnings.filterwarnings("ignore")
        
        print("\n" + "="*70)
        print(f"Avvio CV Gerarchica a {n_iterazioni} Iterazioni: {nome_modello}")
        print(f"Soglia Cestino Stage 2: {soglia_confidenza_s2*100}%")
        print("="*70)
        
        # 1. Definizione delle classi per la valutazione
        hydrocarbons = ['METHANE', 'BUTANE', 'GASOLINE', 'LIGHTER_FLUID', 'DIESEL', 'KEROSENE']
        LABEL_NON_HYDRO = 'NON_HYDROCARBON'
        
        classes_eval = [LABEL_NON_HYDRO] + sorted(hydrocarbons)
        
        all_cms = []
        accuracies, accuracies_s1, accuracies_s2, acc_no_voting = [], [], [], []
        start_time_total = time.time()
        
        for seed in range(n_iterazioni):
            print(f"--> Esecuzione Iterazione {seed + 1}/{n_iterazioni} (random_state={seed})...")
            
            X, Xts, y, yts, exp_ids_ts = self.generate_trainset(
                self.dict_comp, self.exclude, sensors=self.sensors,
                TR=self.TR, TS=self.TS, random_state=seed
            )
            
            # ==========================================
            # STAGE 1: Addestramento Modello Binario (Numerico)
            # ==========================================
            # 1 per Idrocarburi, 0 per Non-Idrocarburi (Richiesto da LightGBM)
            y_train_bin_num = np.where(np.isin(y, hydrocarbons), 1, 0)
            
            pipe_stage1 = make_pipeline(StandardScaler(), clone(model_stage1))
            pipe_stage1.fit(X, y_train_bin_num)
            
            # ==========================================
            # STAGE 2: Addestramento Specialista Idrocarburi
            # ==========================================
            mask_hydro_train = np.isin(y, hydrocarbons)
            X_train_hydro = X[mask_hydro_train]
            y_train_hydro_str = y[mask_hydro_train]
            
            le_hydro = LabelEncoder()
            pipe_stage2 = make_pipeline(StandardScaler(), clone(model_stage2))
            
            if len(X_train_hydro) > 0:
                y_train_hydro_enc = le_hydro.fit_transform(y_train_hydro_str)
                pipe_stage2.fit(X_train_hydro, y_train_hydro_enc)
            
            # ==========================================
            # STAGE 1: PREDIZIONE E VOTO BINARIO
            # ==========================================
            y_pred_bin_num_singolo = pipe_stage1.predict(Xts)
            
            df_voti_s1 = pd.DataFrame({'ID_Esperimento': exp_ids_ts, 'Pred_Bin_Singola': y_pred_bin_num_singolo})
            df_voti_s1['Voto_Binario'] = df_voti_s1.groupby('ID_Esperimento')['Pred_Bin_Singola'].transform(lambda x: x.mode()[0])
            
            # Magia: Riconversione in Stringhe per la diagnostica e lo Stage 2
            y_pred_bin_singolo = np.where(df_voti_s1['Pred_Bin_Singola'].values == 1, 'HYDROCARBON', LABEL_NON_HYDRO)
            y_pred_bin_votato = np.where(df_voti_s1['Voto_Binario'].values == 1, 'HYDROCARBON', LABEL_NON_HYDRO)
            
            y_pred_singolo_final = np.copy(y_pred_bin_singolo)
            y_pred_votato_final = np.copy(y_pred_bin_votato)
            
            # ==========================================
            # STAGE 2: PREDIZIONE A CASCATA CON "CESTINO"
            # ==========================================
            mask_pred_hydro_votato = (y_pred_bin_votato == 'HYDROCARBON')
            
            if np.any(mask_pred_hydro_votato) and len(X_train_hydro) > 0:
                # 1. Predizione delle PROBABILITÀ invece delle classi nette
                proba_hydro = pipe_stage2.predict_proba(Xts[mask_pred_hydro_votato])
                
                # Troviamo la classe più probabile e la sua probabilità (Confidenza)
                max_proba = np.max(proba_hydro, axis=1)
                pred_hydro_enc = np.argmax(proba_hydro, axis=1)
                pred_hydro_str = le_hydro.inverse_transform(pred_hydro_enc)
                
                # --- IL CESTINO ---
                # Se la confidenza è sotto la soglia, declassiamo il campione a NON_HYDROCARBON
                mask_low_conf = max_proba < soglia_confidenza_s2
                pred_hydro_str[mask_low_conf] = LABEL_NON_HYDRO
                
                # Salviamo i risultati singoli
                y_pred_singolo_final[mask_pred_hydro_votato] = pred_hydro_str
                
                # 2. Voto finale: aggreghiamo le predizioni (incluse quelle "cestinate")
                df_voti_s2 = pd.DataFrame({
                    'ID_Esperimento': exp_ids_ts[mask_pred_hydro_votato],
                    'Pred_S2_Singola': pred_hydro_str
                })
                voto_s2 = df_voti_s2.groupby('ID_Esperimento')['Pred_S2_Singola'].transform(lambda x: x.mode()[0])
                
                y_pred_votato_final[mask_pred_hydro_votato] = voto_s2.values
                
            # ==========================================
            # AGGIORNAMENTO METRICHE
            # ==========================================
            y_true_eval = np.where(np.isin(yts, hydrocarbons), yts, LABEL_NON_HYDRO)
            
            acc_singola = accuracy_score(y_true_eval, y_pred_singolo_final)
            acc_no_voting.append(acc_singola)
            
            acc_votata = accuracy_score(y_true_eval, y_pred_votato_final)
            accuracies.append(acc_votata)
            
            # --- Diagnostica Isolata ---
            y_true_binario = np.where(y_true_eval != LABEL_NON_HYDRO, 'HYDROCARBON', LABEL_NON_HYDRO)
            acc_s1 = accuracy_score(y_true_binario, y_pred_bin_votato)
            accuracies_s1.append(acc_s1)
            
            acc_s2 = 0
            mask_veri_idro_promossi = (y_true_binario == 'HYDROCARBON') & (y_pred_bin_votato == 'HYDROCARBON')
            if np.any(mask_veri_idro_promossi):
                acc_s2 = accuracy_score(y_true_eval[mask_veri_idro_promossi], y_pred_votato_final[mask_veri_idro_promossi])
                accuracies_s2.append(acc_s2)
            else:
                accuracies_s2.append(0.0)
                
            print(f"    Accuracy Stage 1: {(acc_s1*100):.2f}% | Accuracy Stage 2: {(acc_s2*100):.2f}%")
            print(f"    Accuracy Singola: {(acc_singola*100):.2f}% | Votata: {(acc_votata*100):.2f}%")
            
            cm = confusion_matrix(y_true_eval, y_pred_votato_final, labels=classes_eval, normalize='true')
            all_cms.append(cm * 100)
            
        # ==========================================
        # AGGREGAZIONE E SALVATAGGIO GRAFICI
        # ==========================================
        mean_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        tempo_totale = time.time() - start_time_total
        print(f"\nValidazione completata in {tempo_totale:.2f} secondi!")
        print(f"Accuratezza Media Stage 1 (Binario): {(np.mean(accuracies_s1)*100):.2f}% ± {(np.std(accuracies_s1)*100):.2f}%")
        print(f"Accuratezza Media Stage 2 (Specialista): {(np.mean(accuracies_s2)*100):.2f}% ± {(np.std(accuracies_s2)*100):.2f}%")
        print(f"Accuratezza Media Votata: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        annot = np.empty_like(mean_cm, dtype=object)
        n = len(classes_eval)
        for i in range(n):
            for j in range(n):
                if mean_cm[i, j] >= 0.1:
                    annot[i, j] = f"{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}"
                else:
                    annot[i, j] = ""

        fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
        import seaborn as sns
        sns.heatmap(mean_cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                    xticklabels=classes_eval, yticklabels=classes_eval, ax=ax, 
                    annot_kws={"size": 9}, linewidths=0.5, linecolor="lightgray")
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        accuracy_string = f"{mean_acc:>5.1f}% ± {std_acc:>4.1f}%"
        titolo_grafico = f"{n_iterazioni}-Fold CV: {nome_modello}\nAccuracy: {accuracy_string}"
        plt.title(titolo_grafico, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel("Etichetta Reale", fontweight='bold', fontsize=12)
        plt.xlabel("Etichetta Predetta", fontweight='bold', fontsize=12)
        
        os.makedirs("Risultati", exist_ok=True)
        nome_file = f"Risultati/{dir}CV_Gerarchica_{nome}_{nome_modello.replace(' ', '_')}_CM.png"
        plt.savefig(nome_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Grafico salvato in: {nome_file}\n")
        
        # Calcolo del tempo formattato
        minuti, secondi = divmod(int(tempo_totale), 60)
        ore, minuti = divmod(minuti, 60)
        tempo_string = f"{ore:02d}:{minuti:02d}:{secondi:02d}"
        
        accuracy_s2_string = f"{(np.mean(accuracies_s2)*100):>5.1f}% ± {(np.std(accuracies_s2)*100):>4.1f}%"
        
        return accuracy_string, accuracy_s2_string, tempo_string
    
    
    def prova_strati_idrocarburi(self):
        
        # 1. IL GUARDIANO INSUPERABILE (Fissato e Intoccabile)
        model_stage1 = LGBMClassifier(
            n_estimators=1000, 
            learning_rate=0.01, 
            num_leaves=63,
            scale_pos_weight=15.0, 
            random_state=42, 
            n_jobs=-1, 
            verbose=-1
        )

        # 2. L'ARSENALE DELLE ARCHITETTURE
        architetture = [
            # 1. ARCHITETTURE "A IMBUTO"
            (256, 128, 64), (128, 64, 32), (64, 32, 16), (32, 16, 8),
            (256, 128, 64, 32, 16), (128, 64, 32, 16), (64, 32, 16, 8), 
            (128, 64, 32, 16, 8), (128, 64), (64, 32), (32, 16), (16, 8),
            # Imbuti "Soft"
            (128, 96, 64, 32, 16), (64, 48, 32, 16, 8),
            # Imbuti con Plateau
            (256, 256, 128), (128, 128, 64), (64, 32, 16, 8, 8),
            (32, 16, 8, 8, 8), (32, 16, 8, 8), (16, 8, 8, 8, 8), 
            (16, 8, 8, 8), (16, 8, 8),
        
            # 2. ARCHITETTURE "A DIAMANTE"
            (128, 256, 128), (64, 256, 64), (64, 128, 64), (32, 64, 32), 
            (16, 32, 16), (8, 16, 8),
            # Diamanti profondi
            (64, 128, 256, 128, 64), (32, 64, 128, 64, 32), (16, 32, 64, 32, 16), 
            (8, 16, 32, 16, 8),
            # Diamanti con Plateau centrale
            (128, 256, 256, 128), (64, 128, 128, 64), (32, 64, 64, 32), 
            (16, 32, 32, 16), (8, 16, 16, 8),
        
            # 3. ARCHITETTURE "WIDE & SHALLOW"
            (1024,), (512,), (512, 256), 
        
            # 4. ARCHITETTURE "A COLLO DI BOTTIGLIA"
            (128, 16, 128), (128, 16, 64, 128), (64, 8, 64),
        
            # 5. ARCHITETTURE "A CILINDRO"
            (128, 128, 128), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8),
        
            # 6. ARCHITETTURE "IN ESPANSIONE"
            (64, 128), (32, 64), (16, 32), (8, 16),
            # Espansioni profonde
            (16, 32, 64, 128, 256), (16, 32, 64, 128), (16, 32, 128),
            (8, 16, 32, 64, 128), (8, 16, 32, 64), (8, 16, 32),
            # Espansioni con Plateau iniziale
            (8, 8, 16, 32, 64), (8, 8, 8, 16, 32), (8, 8, 16, 32), 
            (8, 8, 8, 16), (8, 8, 8, 8, 16), (8, 8, 16)
        ]      
        # I parametri base da mantenere costanti durante il confronto delle architetture
        config_base = {
            'solver': 'adam', 
            'batch_size': 1024, 
            'alpha': 0.01, 
            'learning_rate_init': 0.001, 
            'max_iter': 2000,
            'early_stopping': False,
            'random_state': 42
        }       
        nome_file = "Risultati/Architetture_idrocarburi/Risultati_architetture_idrocarburi.txt"
        totale_modelli = len(architetture)

        print("=== INIZIO TEST ARCHITETTURE ===")
        print(f"Totale reti da addestrare: {totale_modelli}")

        with open(nome_file, "w", encoding="utf-8") as file:
            intestazione = f"Report Test Architetture MLP\n"
            file.write(intestazione + "="*130 + "\n")      
        contatore = 1       
        for arch in architetture:
            nome_modello = f"MLP {arch}"

            model_stage2 = MLPClassifier(hidden_layer_sizes=arch, **config_base)
            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")

            accuratezza_str, accuratezza_s2_str, tempo_str = self.classificatore_idrocarburi(
                model_stage1=model_stage1, model_stage2=model_stage2, nome_modello=nome_modello,
                n_iterazioni=10, soglia_confidenza_s2=0.0,dir='Architetture_idrocarburi/')
            riga_risultato = f"{nome_modello:<30} -> Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}"
            print(f"   Completato! Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}\n")
        
            # Salvataggio immediato
            with open(nome_file, "a", encoding="utf-8") as file:
                file.write(riga_risultato + "\n")

            contatore += 1      
        print("=========================================================================================")
        print(f"=== TEST COMPLETATI ===")

        self.genera_top5_idrocarburi(nome_file, "Risultati/Architetture_idrocarburi/top5_architetture.txt")
        print("Classifica generata nel file: top5_architetture.txt")
        
        
    def prova_configurazioni_idrocarburi(self):
        
        model_stage1 = LGBMClassifier(
            n_estimators=1000, 
            learning_rate=0.01, 
            num_leaves=63,
            scale_pos_weight=15.0, 
            random_state=42, 
            n_jobs=-1, 
            verbose=-1
        )
        
        architetture = [
            (64, 32),
            (64, 32, 16, 8),
            (64, 256, 64),
            (512,)
        ]

        configurazioni = [
            { 'alpha': 0.001, 'activation': 'relu' },
            { 'alpha': 0.01,  'activation': 'relu' },
            { 'alpha': 0.1,   'activation': 'relu' },
            { 'alpha': 0.001, 'activation': 'tanh' },
            { 'alpha': 0.01,  'activation': 'tanh' },
            { 'alpha': 0.1,   'activation': 'tanh' }
        ]

        nome_file = "Risultati/Configurazioni_idrocarburi/Risultati_configurazioni_mlp.txt"
        totale_modelli = len(architetture) * len(configurazioni)

        print("=== INIZIO ESPERIMENTI MLP ===")
        print(f"Totale modelli da testare: {totale_modelli}")
        print(f"I risultati verranno salvati in: {nome_file}\n")

        # Gestione log continuo
        with open(nome_file, "w", encoding="utf-8") as file:
            intestazione = f"Report Unico Esperimenti MLP\n"
            intestazione += "="*130 + "\n"
            file.write(intestazione)

        contatore = 1

        # Loop di addestramento
        for arch in architetture:
            for config in configurazioni:
                nome_modello = (f"MLP {arch} | A:{config['alpha']} | AF:{config['activation']}")
                nome_modello_png = (f"MLP {arch} A={config['alpha']} AF={config['activation']}")

                model_stage2 = MLPClassifier(
                    hidden_layer_sizes=arch,
                    batch_size=1024,
                    max_iter=2000, 
                    early_stopping=False,
                    learning_rate='adaptive',
                    random_state=42,
                    **config
                )

                print(f"[{contatore}/{totale_modelli}] Addestramento in corso: {nome_modello} ...")

                accuratezza_str, accuratezza_s2_str, tempo_str = self.classificatore_idrocarburi(
                model_stage1=model_stage1, model_stage2=model_stage2, nome_modello=nome_modello_png,
                n_iterazioni=10, soglia_confidenza_s2=0.0,dir='Configurazioni_idrocarburi/')
                riga_risultato = f"{nome_modello:<40} -> Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}"
                print(f"   Completato! Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}\n")

                with open(nome_file, "a", encoding="utf-8") as file:
                    file.write(riga_risultato + "\n")

                contatore += 1

        print("=========================================================================================")
        print(f"=== TUTTI GLI ESPERIMENTI SONO CONCLUSI CON SUCCESSO ===")

        # Generazione automatica del file Top 5
        self.genera_top5_idrocarburi(nome_file, "Risultati/Configurazioni_idrocarburi/top5_configurazioni.txt")
    
    
    def prova_soglia_idrocarburi(self):
        
        model_stage1 = LGBMClassifier(
                n_estimators=1000, 
                learning_rate=0.01, 
                num_leaves=63,
                scale_pos_weight=15.0, 
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
            )
        
        model_stage2 = MLPClassifier(
                    hidden_layer_sizes=(64, 256, 64),
                    batch_size=1024,
                    alpha=0.01,
                    max_iter=2000, 
                    early_stopping=False,
                    learning_rate='adaptive',
                    random_state=42,
                )
        
        soglie = [0.4, 0.5, 0.6, 0.7, 0.8]
        
        nome_file = "Risultati/Soglia_idrocarburi/Risultati_soglia_idrocarburi.txt"
        totale_modelli = len(soglie)

        print("=== INIZIO TEST SOGLIE ===")
        print(f"Totale reti da addestrare: {totale_modelli}")

        with open(nome_file, "w", encoding="utf-8") as file:
            intestazione = f"Report Test Soglie MLP\n"
            file.write(intestazione + "="*130 + "\n")      
        contatore = 1       
        for soglia in soglie:
            nome_modello = f"Gerarchico soglia {soglia}"

            print(f"[{contatore}/{totale_modelli}] Addestramento: {nome_modello} ...")

            accuratezza_str, accuratezza_s2_str, tempo_str = self.classificatore_idrocarburi(
                model_stage1=model_stage1, model_stage2=model_stage2, nome_modello=nome_modello,
                n_iterazioni=10, soglia_confidenza_s2=soglia,dir='Soglia_idrocarburi/')
            riga_risultato = f"{nome_modello:<30} -> Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}"
            print(f"   Completato! Accuracy globale: {accuratezza_str} | Accuracy stage 2: {accuratezza_s2_str} | Tempo: {tempo_str}\n")
        
            # Salvataggio immediato
            with open(nome_file, "a", encoding="utf-8") as file:
                file.write(riga_risultato + "\n")

            contatore += 1      
        print("=========================================================================================")
        print(f"=== TEST COMPLETATI ===")
            
    
    def genera_top5_idrocarburi(self, file_log, file_output):
        """Legge il log, estrae le percentuali e salva le prime 5 in un nuovo file."""
        
        if not os.path.exists(file_log):
            print(f"Errore: Il file {file_log} non esiste. Impossibile generare la Top 5.")
            return

        risultati = []
        # La regex cerca "Accuracy: ", poi cattura numeri e punti decimali, seguiti da "%"
        pattern_acc_s2 = re.compile(r"Accuracy stage 2:\s*([0-9\.]+)%")

        with open(file_log, "r", encoding="utf-8") as f:
            for linea in f:
                match = pattern_acc_s2.search(linea)
                if match:
                    # Estraiamo il valore float per l'ordinamento matematico
                    acc_s2_val = float(match.group(1))
                    risultati.append((acc_s2_val, linea.strip()))

        # Ordina la lista basandosi sul primo elemento (l'accuratezza Stage 2) in ordine decrescente
        risultati_ordinati = sorted(risultati, key=lambda x: x[0], reverse=True)
        top5 = risultati_ordinati[:5]

        # Scrive la classifica nel file output
        with open(file_output, "w", encoding="utf-8") as f:
            intestazione = f"=== TOP 5 CONFIGURAZIONI MLP (Ordinate per Stage 2) ===\n"
            f.write(intestazione + "="*130 + "\n")

            for i, (_, riga) in enumerate(top5, 1):
                f.write(f"{i}° Posto | {riga}\n")

        print(f"\nClassifica 'Top 5' generata con successo nel file: {file_output}")
    
    
    
#Gestore centrale del dataset, prepara e organizza i dati
class SCA_NRM_data:
    #Inizializza l'oggetto impostando i percorsi dei file, definendo la lista delle sostanze chimiche da analizzare e caricando i dati in memoria
    #Gestisce un sistema di caching (file .npy), popola le strutture dati interne che mappano ogni sostanza ai suoi esperimenti
    #Opzionalmente esegue una riduzione dei dati e stampa una guida
    def __init__(self,ndeg=-1,sensors=np.arange(16),verbose=0,help=True):
        '''
        :param ndeg: grado delle interpolanti polinomiali usate per ridurre i dati
        :param sensors: array contenente gli indici dei sensori che si vuole includere nel training set
        :param verbose: livello di dettaglio dei log
        :param help: se True, stampa le istruzioni di avvio
        '''
        # traindir: percorso completo alla cartella contenente i dati di training normalizzati
        #           normalmente contenuti nella cartella .../SCA/TRAINING/NORMALIZED/
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.traindir = os.path.join(base_dir, 'TRAINING', 'NORMALIZED') + os.sep
        # components: nome delle sostanze così come compaiono nella cartella traindir
        self.components = ['ACETIC_ACID', 'AIR', 'ACETONE', 'AMMONIUM_CHLORIDE', 'AMMONIA', 'CALCIUM_NITRATE', 'BUTANE',
                           'BIOETHANOL', 'APPLE_VINEGAR', 'DIESEL', 'GASOLINE', 'FORMIC_ACID', 'ETHANOL', 'ISOPROPANOL',
                           'HYDROGEN_PEROXIDE', 'METHANE', 'LIGHTER_FLUID', 'KEROSENE', 'RED_WINE', 'PHOSPHORIC_ACID',
                           'NITROMETHANE', 'WATER_VAPOR', 'UREA', 'SODIUM_HYDROXIDE', 'BALSAMIC_VINEGAR']
        # filenamepickle: nome del file pickle che si vuole usare per salvare i dati oppure (se esistente) per leggere i dati
        self.filenamepickle = "SCA_NRM_DATA2025-06-05.npy"

        #Controlla se il file di cache esiste
        if os.path.isfile(self.filenamepickle):
            #Se esiste i dati vengono caricati direttamente dal file
            self.extract_data(export=False)
        else:
            #Altrimenti legge tutti i dati dalla cartella traindir e crea il file .npy per il futuro
            self.extract_data(export=True)

        #Crea un dizionario per dividere gli esperimenti in base alla sostanza (componente, lista_esperimenti)
        self.experiments = {}
        #Itera ogni composto nella lista creata prima
        for component in self.components:
            #Inizializza un array di esperimenti effettuati su quel composto (al momento vuoto)
            expr = []
            #Itera su ogni coppia (componente, id_esperimento) presente nel dizionario principale dei dati
            for (a,b) in self.DATA.keys():
                #Se il componente è uguale a quello che stiamo iterando allora aggiunge id_esperimento alla lista expr
                if a == component:
                    expr.append(b)
            #Aggiunge al dizionario la lista di esperimenti per quel composto
            self.experiments[component] = expr

        #Stampa opzionalmente un report sulle proprietà dei dati caricati
        if verbose > 0:
            self.data_prop()
        #Memorizza i parametri del costruttore
        self.ndeg = ndeg
        self.sensors = sensors

        #Stampa opzionalmente a video una spiegazione sulla struttura dei dati e come accedervi
        if help:
            self.help()
                
    #Gestisce il caricamento dei dati nel dizionario
    def extract_data(self,export=True):
        '''
        procedura che legge i dati dei sensori SCA normalizzati.
        Se export=True,  i dati sono letti dai file csv inviati da SENSICHIPS e quindi salvati nel file SCA_NRM_DATA.npy
        Se export=False, i dati sono letti direttamente dal file SCA_NRM_DATA.npy precedentemente creato
        Se non si ha il file SCA_NRM_DATA.npy oppure lo si è cancellato bisogna usare export=True
        '''
        #Crea il dizionario vuoto che conterrà tutto il dataset
        self.DATA = {}

        #Se abbiamo bisogno di creare il file .npy
        if export:
            #Itera i componenti
            for component in self.components:
                #Logging
                print(component+' ',end='')
                #Costruisce il percorso della cartella corrispondente concatenando la directory base e il nome del componente
                traindata = self.traindir + component

                #Se la cartella esiste
                if os.path.isdir(traindata):
                    #Inizializza un contatore a 1 per numerare gli esperimenti trovati per quella sostanza (ID_Esperimento)
                    expid = 1
                    #Itera tutti i file nella cartella
                    for file in os.listdir(traindata):
                        #Se il file è .csv
                        if file.endswith(".csv"):
                            #Stampa l'ID dell'esperimento che sta leggendo
                            print('exp'+str(expid)+' ',end='')
                            #Usa pd.read_csv per leggere il file e lo salva nel dizionario insieme al nome del file
                            self.DATA[(component,expid)] = (pd.read_csv(traindata + '/' + file, sep=';'),file)
                            #Incrementa l'ID del prossimo file
                            expid += 1
                print()

            #Apre il file binario di destinazione in modalità scrittura binaria (wb)
            filepickle = open(self.filenamepickle,'wb')
            #Salva l'intero oggetto self.Data sul disco
            pickle.dump(self.DATA,filepickle)
            #Chiude il file
            filepickle.close()
        #Altrimenti apre il file binario in modalità lettura binaria (rb)
        else:
            with open(self.filenamepickle, 'rb') as file:
                #Ricostruisce l'intero dizionario self.DATA in memoria
                self.DATA = pickle.load(file)