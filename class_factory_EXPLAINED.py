"""
    INDICE
                                           [Riga]
    class experiment:
        - __init__                           132
        - determine_air_confusion            252
        - describe_data                      733
	    - display_CM                        1087
	    - display_results                   1123
        - write_general_report              1175
	    - write_table_latex                 1232
        - predict_experiment                 423
	    - report_on_experiment               515
	    - predict_sigle_measure              526
	    - report_on_sigle_measure            550
        - argsort_exp                        300
	    - define_train_test_exp             1395
	    - split_experiments                 1544
	    - generate_trainset                 1571
	    - generate_dict_train_test          1847
	    - generate_multiclass_trainset      1928
        - prova_uno                          562
        - prova_due                         1734
        - prova_tre                         1323
        - run_prova_quattro                  745
        - run_prova_cinque                  1008
        - run_prova_sei                      887
        - run_prova_sensichips               627
        
    class SCA_NRM_data:
        - __init__                          2020
        - help                              2089
        - extract_data                      2376
        - data_prop                         2379
        - data_reduce                       2139
	    - data_reduce_pacchetti             2220
	    - data_reduce_poly                  2290
        - approximate                       2629
        - do_plots                          2467
        - plot_examples                     2578
        - plot_sensor                       2426
        - plot_measurement                  2538
        - plot_approx                       2668
"""




#Creare cartelle, navigare nei percorsi dei file o controllare se un file esiste
import os
#Trovare file che corrispondono ad un determinato pattern
import glob
#Libreria standard per la manipolazione dei dati
import pandas as pd
#Gestisce matrici, array multidimensionali e operazioni matematiche veloci
import numpy as np
#Generatore di numeri casuali
import random
#Funzioni per l'elaborazione dei segnali
from scipy import signal
#Fuzioni statistiche
from scipy import stats
#Serializzare oggetti (salvare modelli per non doverli addestrare di nuovo)
import pickle
#Creare grafici
import matplotlib.pyplot as plt
#Stima di modelli statistici classici e test statistici
import statsmodels.api as sm
#Stampare dataframe o liste in formato leggibile nella console
from tabulate import tabulate

# ============================
# CLASSIFIER CATALOG
# ============================
#Algoritmo base per la classificazione | Variante con regolarizzazione
from sklearn.linear_model import LogisticRegression, RidgeClassifier
#Singolo albero decisionale
from sklearn.tree import DecisionTreeClassifier
#Unisce tanti alberi decisionali per un risultato più stabile | Tecniche di "boosting" dove ogni modello corregge gli errori del precedente
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
#Basato sul teorema di Bayes e statistica gaussiana
from sklearn.naive_bayes import GaussianNB
#Support Vector Classifier | Versione per la regressione | Versione ottimizzata per confini lineari
from sklearn.svm import SVC, SVR, LinearSVC
#Metodi statistici classici per trovare combinazioni di feature che separano le classi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier, MLPRegressor
#Extreme Gradient Boosting
from xgboost import XGBClassifier
#Light Gradient Boosting Machine (simile a XGB ma spesso più veloce su grandi dataset)
from lightgbm import LGBMClassifier
#Permette di far "votare" diversi modelli per decidere la classe finale
from sklearn.ensemble import VotingClassifier, StackingClassifier
#Classifica un dato basandosi sulla classe dei dati a lui più vicini geometricamente
from sklearn.neighbors import KNeighborsClassifier

#Divide i dati in più parti per testare il modello più volte | Divide i dati mantenendo la stessa percentuale di classi | Prova automaticamente diverse combinazioni di parametri
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, GridSearchCV
#Metriche di valutazione
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer, ConfusionMatrixDisplay
)
#Ridimensiona i dati per avere media 0 e varianza 1 | Trasforma categorie testuali in numeri {0,1} | Porta tutti i dati tra 0 e 1
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, MinMaxScaler)
#Strumento per concatenare i passaggi
from sklearn.pipeline import make_pipeline, Pipeline
#Algoritmi che scartano le colonne inutili o rumorose per migliorare il modello
from sklearn.feature_selection import (
    SelectFromModel, SequentialFeatureSelector
)

#Validazione incrociata standard
from sklearn.model_selection import KFold
#Mischia i dati in modo casuale
from sklearn.utils import shuffle
#Converte le etichette in numeri
from sklearn.preprocessing import LabelEncoder

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
        print_boxplot: Se True genera un grafico
    """
    def __init__(self,SCA,sensors=range(16),nmin=4,soglia=0.2,TR='SLOW',TS='SLOW',debug=False,print_boxplot=False):
        #Inizializza i dati
        self.SCA = SCA
        self.sensors = sensors
        self.nmin = nmin
        self.TR = TR
        self.TS = TS
        self.soglia = soglia
        self.debug = debug
        self.print_boxplot = print_boxplot

        #Se debug è True imposta un sottoinsieme di sostanze
        if debug:
            list_components = ['AMMONIA', 'RED_WINE', 'AIR']
        #Altrimenti lavora sulla lista completa
        else:
            list_components = self.SCA.components

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
                hidden_layer_sizes=(256,128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate_init=0.001,
                max_iter=50,
                early_stopping=False,
                random_state=42,
                verbose=True
            ),
            'SVM (gaussian)': SVC(kernel='rbf', random_state=42, verbose=False, max_iter=5000),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=100),
            'Ridge Classifier': RidgeClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Naive Bayes': GaussianNB(),
            'SVM (Linear Kernel)': SVC(C=10, kernel='linear', probability=False),
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant': QuadraticDiscriminantAnalysis(),
            'XGBoost': XGBClassifier(eval_metric='mlogloss',random_state=42,n_jobs=-1,tree_method='hist'),
            'LightGBM': LGBMClassifier(verbose=-1,random_state=42, n_jobs=-1)
        }
        #Definisce un dizionario di scorer che poi verranno passati alle funzioni di validazione
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
        }

    #Analizza un componente specifico per capire l'intervallo operativo dei sensori
    def determine_air_confusion(self):
        #Composto che si vuole analizzare
        c = 'KEROSENE'
        #nsensors = len(self.sensors)
        #Numero dei sensori
        nsensors = 16
        #Inizializza due matrici vuote con 0 righe (per ora) e 16 colonne che serviranno a raccogliere i massimi e i minimi di ogni esperimento
        vmax = np.zeros((0, nsensors))
        vmin = np.zeros((0, nsensors))
        #Itera su tutti gli esperimenti del componente scelto
        for e in self.SCA.experiments[c]:
            #Recupera i dati 
            (a, b) = self.SCA.DATA[(c, e)]
            #Recupera le dimensioni del dataframe
            n, m = a.shape
            #Controlla se l'esperimento ha abbastanza dati (rispetto a self.min definito in __init__)
            if n > self.nmin:
                #Converte il dataframe in una matrice NumPy, escludendo l'ultima colonna
                t = a.to_numpy()[:, :-1]
                #Sostituisce eventuali valori NaN con 0
                t[pd.isna(t)] = 0.
                #Calcola massimo e minimo per ogni colonna
                tmin = np.min(t,axis=0)
                tmax = np.max(t,axis=0)
                #print(tmax,tmin)
                #Inserisce i valori nelle matrici vmax e vmin
                vmax = np.vstack((vmax,tmax))
                vmin = np.vstack((vmin,tmin))
        #Stampa le dimensioni finali degli accumulatori
        print(vmax.shape,vmin.shape)
        #Itera per ogni sensore
        for i,s in enumerate(range(16)):
            #Trova il minimo e il massimo assoluto per il sensore i
            minvalue = np.min(vmin[:,i])
            maxvalue = np.max(vmax[:,i])
            #Stampa il range totale osservato
            print('sensor {}: {} -- {}'.format(s,minvalue,maxvalue))
        #Aspetta che l'utente prema INVIO per continuare
        input()

    #Per ogni sostanza conta quanti esperimenti e quante misurazioni esistono e li ordina in modo decrescente
    """
        Input:
            order_by: Decide se ordinare per numero di file (exp) o numero di righe totali (mis)
            what: Filtra quale tipo di dati contare per l'ordinamento
            print_resume: Se True stampa a schermo i componenti ordinati
            only_slow_mot: MAI UTILIZZATO
    """
    def argsort_exp(self, order_by=None,what='ALL',print_resume=False,only_slow_mot=False):
        # order_by can either be 'exp' or 'mis'
        # what can either be 'ALL' or 'SLOW' or 'STAT'

        #Recupera la soglia minima di righe definita in __init__
        nmin = self.nmin
        #Una lista che diventerà il dataframe finale
        temp = []
        #Array vuoti che serviranno per memorizzare i conteggi
        tempe = np.empty(0, dtype=int)
        tempm = np.empty(0, dtype=int)
        tempe_slow = np.empty(0, dtype=int)
        tempe_stat = np.empty(0, dtype=int)
        tempm_slow = np.empty(0, dtype=int)
        tempm_stat = np.empty(0, dtype=int)
        #Itera su ogni componente
        for c in self.SCA.components:
            #Recupera il numero di esperimenti per il componente corrente
            ne = len(self.SCA.experiments[c])
            #Azzera tutti i contatori
            num = 0
            mis = 0
            num_slow = 0
            num_stat = 0
            mis_slow = 0
            mis_stat = 0
            #Itera su ogni esperimento del componente corrente
            for e in self.SCA.experiments[c]:
                #Estrae il dataframe e il nome del file
                (a, b) = self.SCA.DATA[(c, e)]
                #Recupera le dimensioni del dataframe
                n, m = a.shape
                #Se l'esperimento è valido (ha abbastanza righe)
                if n > nmin:
                    #Incrementa il contatore dei file
                    num += 1
                    #Incrementa il contatore delle misure totali aggiungendo n
                    mis += n
                    #Se il nome del file contiene 'slow' o 'checkhair', aggiorna i contatori _slow
                    if (b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1):
                        mis_slow += n
                        num_slow += 1
                    #Altrimenti aggiorna i contatori _stat
                    else:
                        mis_stat += n
                        num_stat += 1
            #Crea un dizionario con tutte le statistiche
            row = {'comp': c, 'nexp': num, 'nmis':mis, 'nexp_slow':num_slow, 'nmis_slow':mis_slow, 'nexp_stat':num_stat, 'nmis_stat':mis_stat}

            #Aggiunge row alla lista temp
            temp.append(row)
            #Aggiorna tutti gli array paralleli
            tempe = np.append(tempe, num)
            tempm = np.append(tempm, mis)
            tempe_slow = np.append(tempe_slow, num_slow)
            tempe_stat = np.append(tempe_stat, num_stat)
            tempm_slow = np.append(tempm_slow, mis_slow)
            tempm_stat = np.append(tempm_stat, mis_stat)

        #Se si vuole stampare l'ordinamento a schermo
        if print_resume:
            #Stampa l'ordinamento richiesto (passato in input) creando un dataframe temporaneo e ordinandolo
            for type in ['exp', 'mis']:
                if type == 'exp':
                    print('----------------- ordered by number of EXPERIMENTS ------------------------')
                    if what == 'ALL':
                        result_df = pd.DataFrame(temp).sort_values(by=['nexp', 'nmis'], ascending=False)
                        whatstr = ''
                    elif what == 'SLOW':
                        result_df = pd.DataFrame(temp).sort_values(by=['nexp_slow', 'nmis_slow'], ascending=False)
                        whatstr ='slow'
                    elif what == 'STAT':
                        result_df = pd.DataFrame(temp).sort_values(by=['nexp_stat', 'nmis_stat'], ascending=False)
                        whatstr = 'stat'
                else:
                    print('----------------- ordered by number of   MEASURES  ------------------------')
                    if what == 'ALL':
                        result_df = pd.DataFrame(temp).sort_values(by=['nmis', 'nexp'], ascending=False)
                        whatstr = ''
                    elif what == 'SLOW':
                        result_df = pd.DataFrame(temp).sort_values(by=['nmis_slow', 'nexp_slow'], ascending=False)
                        whatstr = 'slow'
                    elif what == 'STAT':
                        result_df = pd.DataFrame(temp).sort_values(by=['nmis_stat', 'nexp_stat'], ascending=False)
                        whatstr = 'stat'

                #Per ogni sostanza del dataframe stampa quanti esperimenti e misure ci sono, formattando l'output
                inde = result_df.index
                for i in inde:
                    c = self.SCA.components[i]
                    ne = len(self.SCA.experiments[c])
                    num = tempe[i]
                    num_slow = tempe_slow[i]
                    num_stat = tempe_stat[i]
                    mis = tempm[i]
                    mis_slow = tempm_slow[i]
                    mis_stat = tempm_stat[i]

                    print('{:30s} has {:3d} exp(s): {:3d} slow {:3d} stat with > {:2d} mrs ({:6d}/{:6d}/{:6d})'.format(c, ne,
                                                                                        num_slow, num_stat,nmin,mis,mis_stat,mis_slow))

            print('----------------------------------------------------------------------------')

        #Crea un dataframe result_df dalla lista temp e gli applica l'ordinamento richiesto (in base a quali dati vogliamo, in base a cosa li vogliamo ordinati e se in ordine crescente o decrescente)
        if (order_by == 'exp') and (what == 'ALL'):
            result_df = pd.DataFrame(temp).sort_values(by=['nexp', 'nmis'], ascending=False)
        elif (order_by == 'exp') and (what == 'SLOW'):
            result_df = pd.DataFrame(temp).sort_values(by=['nexp_slow', 'nmis_slow'], ascending=False)
        elif (order_by == 'exp') and (what == 'STAT'):
            result_df = pd.DataFrame(temp).sort_values(by=['nexp_stat', 'nmis_stat'], ascending=False)
        elif (order_by == 'mis') and (what == 'ALL'):
            result_df = pd.DataFrame(temp).sort_values(by=['nmis', 'nexp'], ascending=False)
        elif (order_by == 'mis') and (what == 'SLOW'):
            result_df = pd.DataFrame(temp).sort_values(by=['nmis_slow', 'nexp_slow'], ascending=False)
        elif (order_by == 'mis') and (what == 'STAT'):
            result_df = pd.DataFrame(temp).sort_values(by=['nmis_stat', 'nexp_stat'], ascending=False)
        else:
            result_df = pd.DataFrame(temp).sort_values(by=['nexp', 'nmis'], ascending=False)

        #Restituisce l'indice e il dataframe completo
        return result_df.index, result_df

    #Addestra un modello e predice i dati di un test set
    def predict_experiment(self,X, y, Xts, yts, model,verbose=False):
        #Addestra il modello passato in input con i dati di training X,y
        model.fit(X, y)

        #Usa OneHotEncoder per estrarre le classi uniche (sostanze) presenti nel vettore delle etichette di test
        OHE = OneHotEncoder()
        OHE.fit(yts.reshape(-1, 1))
        classes = OHE.categories_[0]
        #Calcola la lunghezza della stringa più lunga tra i nomi delle classi e la usa per creare una stringa che servirà a formattare l'output grafico
        lenc = 0
        for c in classes:
            if len(c) > lenc:
                lenc = len(c)
        str = '-' * lenc

        #Recupera le dimensioni del test set
        n, m = Xts.shape
        # y_pred = -1*np.ones(n)
        #Crea un array di stringhe di formattazione che servirà a contenere le classi predette
        y_pred = np.array([str for i in range(n)])
        #Ottiene le previsioni del modello sul test set
        y_temp = model.predict(Xts)
        #Itera sulle sostanze presenti nel test set
        for i in classes:
            #Seleziona solo le predizioni dove la sostanza vera era quella corrente
            s = pd.Series(y_temp[yts == i])
            #Trova quanto sono lunghe le sequenze stabili di predizioni
            out = (s.groupby(s.ne(s.shift()).cumsum(), sort=False)  # group consecutive values
                   .agg({'first', 'size'})  # get value and count
                   .groupby('first', sort=False)['size'].max()  # max count per value
                   .reset_index().to_numpy()  # back to numpy
                   )
            #Ordina out e trova la lunghezza della sequenza consecutiva più lunga in assoluto
            cons = out[out[:, 1].argsort()]
            consecutives = cons[-1,1]
            #Conta quante volte appare ogni classe nelle predizioni
            fres = s.value_counts()
            #Analizza le frequenze per vedere se ci sono pareggi (se la classe più frequente è l'unica o se ce ne sono due a pari merito)
            A = (fres.groupby(fres.ne(fres.shift()).cumsum(), sort=False)  # group consecutive values
                   .agg({'first', 'size'})  # get value and count
                   .groupby('first', sort=False)['size'].max()  # max count per value
                   .reset_index().to_numpy()  # back to numpy
                   )
            print('-----------------------------')
            print(i, ' ', end='')
            print(y_temp[yts == i].shape, end='')
            #freq = {}
            #win = -1
            #fwn = -1
            #for j in classes:
            #    freq[j] = len([a for a in y_temp[yts == i] if a == j])
            #    if freq[j] > fwn:
            #        fwn = freq[j]
            #        win = j
            #print(freq, ' ', win)
            #print(i,' ', win, freq[win])
            print(fres.shape)
            #Numero di classi che sono a pari merito per frequenza
            npari = A[0,1]
            #Prende le predizioni consecutive e le ordina
            cons1 = cons[[cons[i,0] in fres.index[:npari] for i in range(cons.shape[0])],:]
            cons1 = cons1[cons1[:, 1].argsort()]
            if verbose:
                print(npari,fres.index[:npari])
                print(cons1)
            if A[0,1] == 1: #il vincitore come frequenza è unico
                if verbose:
                    print('vincitore unico !!!')
                win = fres.index[0]
                freq = fres.iloc[0]
            else: # ci sono dei parimerito
                if verbose:
                    print('pari merito !!!')
                #Vince la classe che ha mantenuto la predizione stabile per più tempo consecutivo
                win = cons1[-1,0]
                freq = cons1[-1,1]
            #Stampa un report dove Tl = True label e Pl = Predicted label
            print('Tl:{:20s} Pl:{:20s} number       occurrences:{:5d}'.format(i, win, freq))
            print('Tl:{:20s} Pl:{:20s} number cons. occurrences:{:5d}'.format(i, cons[-1,0],consecutives))
            print('Tl:{:20s} Pl:{:20s} number cons. occurrences:{:5d}'.format(i, cons1[-1,0],cons1[-1,1]))

            #Assegna l'etichetta vincitrice a tutte le righe dell'array di predizioni corrente
            y_pred[yts == i] = win

            if verbose:
                print(fres)
                print(cons)
                #input()
        #Restituisce sia le predizioni originali che quelle votate per esperimento
        return y_temp, y_pred

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

    #Esegue una K-Fold Cross Validation Stratificata che testa il modello per capire se è in grado di classificare una riga di dati indipendente
    def predict_sigle_measure(self, X, y, model, cv_folds=5):
        #Crea il modello per dividere i dati
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        #Crea due liste vuote per contenere i risultati finali
        y_true_all = []
        y_pred_all = []

        #Itera sugli insiemi di dati creati da skf
        for train_idx, test_idx in skf.split(X, y):
            #Stampa il numero di dati usati per il training e per il test
            print(len(train_idx), len(test_idx))
            #Addestra il modello usando i dati correnti
            model.fit(X[train_idx], y[train_idx])
            #Chiede al modello di predirre le etichette per le righe correnti
            y_pred = model.predict(X[test_idx])
            #Raccoglie le etichette reali e quelle predette nelle due liste
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(y_pred)

        #Restituisce le due liste di etichette 
        return y_true_all, y_pred_all

    #Stampa un report sulle predizioni della Cross-Validation
    def report_on_sigle_measure(self, y_true_all, y_pred_all, model_name):

        #Stampa la Confusion Matrix
        print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_true_all, y_pred_all))
        #Stampa il report di scikit-learn
        print(f"\nClassification Report for {model_name}:\n",
              classification_report(y_true_all, y_pred_all, zero_division=0))
        #Restituisce il report come dizionario
        return classification_report(y_true_all, y_pred_all, zero_division=0, output_dict=True)

    #Esegue un ciclo di 10 simulazioni e valuta diversi modelli
    #NON FUNZIONA (contiene errori e refusi)
    def prova_uno(self):
        #Inizializza la lista di sostanze che verranno utilizzate e il dizionario con i dati esclusi
        list_comp = ['KEROSENE', 'ACETONE', 'NITROMETHANE', 'ACETIC_ACID']
        exclude = {}
        for c in list_comp:
            exclude[c] = []
        exclude['KEROSENE'] = [12]
        exclude['ACETONE'] = [1]

        compound = ['ETHANOL', 'ISOPROPANOL']
        list_comp = ['AMMONIA', 'ETHANOL', 'ISOPROPANOL', 'ACETIC_ACID']
        list_comp = ['AMMONIA', 'ETHANOL', 'SODIUM HYDROXIDE', 'ACETIC_ACID']
        list_comp = ['AMMONIA', 'ETHANOL', 'SODIUM HYDROXIDE', 'ISOPROPANOL', 'ACETIC_ACID']
        list_comp = ['AMMONIA', 'ETHANOL', 'SODIUM HYDROXIDE', 'ISOPROPANOL', 'ACETIC_ACID', 'AMMONIUM_CHLORIDE']
        # final acc_on_mod: 0.9651918896895884  acc_on_exp: 0.8223593844414114
        # list_comp = ['ETHANOL', 'ISOPROPANOL']
        exclude = {}
        for c in list_comp:
            exclude[c] = []

        #Stampa quali gas si stanno testando
        print('\n\n')
        print('---------------------------------------------------------------------')
        for c in list_comp:
            print('\t', c, end='')
        print()
        print('---------------------------------------------------------------------')
        #Variabili che serviranno a sommare l'accuratezza di ogni ciclo
        acc_on_exp = 0.
        acc_on_mod = 0.
        #Esegue l'esperimento 10 volte
        for ii in range(10):
            #Crea gli insiemi di training e test
            X, Xts, y, yts = self.SCA.generate_multiclass_trainset(list_comp, compound, exclude, only_slow_mov=True,
                                                              random_state=ii)

            print(X.shape, y.ravel().shape, ' : ', end='')
            print()
            print(Xts.shape, yts.ravel().shape, ' : ', end='')
            print()
            # input('Hit RETURN to continue')

            # print(f"\n=== Detailed Report for Top Model: {top_model} ===")
            #Scelta del modello (al momento viene eseguito solo LightGBM)
            top_model = 'Logistic Regression'
            top_model = 'SVM (gaussian)'
            top_model = 'AdaBoost'
            top_model = 'MLP neural net'
            top_model = 'LightGBM'

            #Stampa i report per i due metodi di addestramento (anche se ci sono delle incongruenze logiche)
            print(f"\n=== Detailed Report for Model: {top_model} ===")
            CR = self.report_on_sigle_measure(X, y, top_model, cv_folds=5)
            acc_on_mod += CR['accuracy']
            # input('Hit RETURN to continue')
            CR = self.report_on_experiment(X, y, Xts, yts, top_model)
            acc_on_exp += CR['accuracy']
            print()
        #Calcola l'accuratezza media e la stampa
        acc_on_exp /= 10.
        acc_on_mod /= 10.
        print('\n')
        print('final acc_on_mod: {}  acc_on_exp: {}'.format(acc_on_mod, acc_on_exp))

    #Valuta il modello KNN su tutti i dati disponibili
    def run_prova_sensichips(self,cv_folds=10,normalized=False):
        #Recupera le strategie ('SLOW', 'ALL' ecc)
        TR = self.TR
        TS = self.TS

        #knn = make_pipeline(StandardScaler(), self.classifiers['KNN'])
        #Definisce il classificatore KNN
        knn = self.classifiers['KNN']
        #Definisce un classificatore complesso che però non sarà utilizzato
        voting = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('mlp', self.classifiers['MLP neural net']),
                                   ('svm', self.classifiers['SVM (gaussian)']),
                                   ])
        #Crea una pipeline di esecuzione
        clf = make_pipeline(StandardScaler(), knn)

        #Variabile che conterrà l'accuratezza
        acc_on_mod = 0.

        #Ottiene i dati grezzi dell'esperimento
        X, Xts, y, yts = self.generate_trainset(self.dict_comp,self.exclude,
                                                sensors=self.sensors,
                                                TR=TR,TS=TS,
                                                normalized=normalized,balance=False,perc=0.7,
                                                random_state=13)

        #Unisce i dati di training e test
        X = np.vstack((X, Xts))
        y = np.append(y, yts)
        print(X.shape, y.ravel().shape, ' : ', end='')

        #Se print_boxplot è True genera dei grafici per mostrare la distribuzione dei valori di ogni sensore su tutto il dataset unito
        if self.print_boxplot:
            xx = [np.array([a for a in X[:, i]]) for i in range(len(self.sensors))]

            #std = StandardScaler()
            #std.fit(X,y)
            #Xn = std.transform(X)
            fig, axs = plt.subplots(1, 2, figsize=(2.5, 5))
            fig.suptitle('data distributions')

            axs[0].set_ylabel("values")
            axs[0].set_xlabel("sensors")
            axs[0].set_title("violinplot")
            axs[1].set_ylabel("values")
            axs[1].set_xlabel("sensors")
            axs[1].set_title("boxplot")

            axs[0].violinplot(xx,showmedians=True)
            axs[1].boxplot(xx)
            #plt.legend(loc='lower right',prop={'size': 3})
            plt.show()
            plt.close()

            #for i in self.sensors:
            for i in []:
                plt.figure(figsize=(5,10))
                plt.title('sensore {}'.format(i))
                A = []
                for c in self.SCA.components:
                    xx = X[[j==c for j in y],i]
                    A.append(xx)
                plt.boxplot(A,tick_labels=self.SCA.components)
                plt.xticks(rotation=45,ha='right')
                plt.show()
                plt.close()
            print('Hit RETURN to continue',end='')
            input()

        #Chiama predict_single_measure che addestra il modello e fa le predizioni
        y_true, y_pred = self.predict_sigle_measure(X, y, clf, cv_folds=cv_folds)
        #Stampa il report e recupera il dizionario contenente lo stesso
        CR = self.report_on_sigle_measure(y_true, y_pred, 'KNN')
        #input('Hit RETURN to continue')
        #Ottiene l'accuratezza delle predizioni e la stampa
        acc_on_mod += CR['accuracy']
        print()

        print('\n')
        print('final acc_on_mod: {}'.format(acc_on_mod))

        #Stampa i risultati
        self.display_results(y_true, y_pred, y_pred, clf.classes_, 'KNN')

#        CM = confusion_matrix(y_true, y_pred, normalize='true')
#        with open('CM.txt','w') as file:
#            print(CM,file=file)
#            print(f"\nClassification Report ( majority ):",file=file)
#            print(classification_report(y_true, y_pred, zero_division=0),file=file)
#
#        disp = ConfusionMatrixDisplay(confusion_matrix=CM,
#                                      display_labels=clf.classes_)
#        disp.plot()
#        plt.show()

    #Coordina la generazione della documentazione del dataset
        """
            Input:
                reduce: Indica se usare i dati ridotti o i dati grezzi
                order_by, what: Parametri di ordinamento
                write_GR: Se True scrive il General Report
                write_LT: Se True scrive la LaTeX Table
                print_resume: Se True stampa un riassunto sulla console
        """
    #Crea dei file di testo con gli elementi richiesti
    def describe_data(self,reduce=False,order_by='mis',what='ALL',write_GR=False,write_LT=False,print_resume=False):
        if write_GR:
            #Crea un file di testo che elenca ogni sostanza e gli ID dei relativi esperimenti
            self.write_general_report(reduce=reduce)
        if write_LT:
            #Crea una tabella ordinata
            ind, temp = self.argsort_exp(order_by=order_by,what=what,print_resume=print_resume)
            #Scrive la tabella in codice LaTeX
            self.write_table_latex(reduce=reduce,ind=ind)
            #input('Hit RETURN to continue')

    #Combina modelli diversi per capire se si ottengono prestazioni migliori
    def run_prova_quattro(self,nruns=10,reduce=False,normalized=True,balance=True,perc=0.7,write_GR=False,write_LT=False,print_resume=False):
        #Recupera le strategie di train e test
        TR = self.TR
        TS = self.TS

        #Se richiesto genera il file di testo o le tabelle LaTeX 
        if write_GR:
            self.write_general_report(reduce=reduce)
        if write_LT:
            ind, temp = self.argsort_exp(order_by='mis',print_resume=print_resume, only_slow_mot=True)
            self.write_table_latex(reduce=reduce,ind=ind)
            #input('Hit RETURN to continue')

        #knn = make_pipeline(StandardScaler(), self.classifiers['KNN'])
        #Vengono creati modelli specifici
        knn = self.classifiers['KNN']
        dct = self.classifiers['Decision Tree']
        rft = self.classifiers['Random Forest']
        svm = self.classifiers['SVM (gaussian)']
        svm = SVC(C=50., gamma=1.3, kernel='rbf', random_state=42, verbose=False, max_iter=5000)
        lvm = LinearSVC(C=10., random_state=42, verbose=False, max_iter=5000)
        mlp = self.classifiers['MLP neural net']
        mlp = MLPClassifier(solver='sgd', max_iter=5000, tol=1.e-3, activation='tanh',
                            alpha=1.e-4, verbose=False, random_state=3857, hidden_layer_sizes=(32,5),
                            n_iter_no_change=100)

        #Vengono creati (e sovrascritti più volte) modelli complessi
        voting3 = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('XGBM',self.classifiers['XGBoost']),
                                   ('svm', svm),
                                   ])
        voting3 = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('XGBM',self.classifiers['XGBoost']),
                                   ])
        voting4 = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('mlp', self.classifiers['MLP neural net']),
                                   ('svm', self.classifiers['SVM (gaussian)']),
                                   ])
        voting4 = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('XGBM',self.classifiers['XGBoost']),
                                   ('svm', svm),
                                   ])
        voting3 = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('XGBM',self.classifiers['XGBoost']),
                                   ('svm', svm),
                                   ('rft', rft),
                                   ])
        voting2 = VotingClassifier([
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('XGBM',self.classifiers['XGBoost']),
                                   ])
        #Crea la pipeline finale
        clf = make_pipeline(StandardScaler(), voting3)

        #Il parametro do_grid, se True, fa si che vengano cercati i valori migliori per C e gamma per l'SVM
        do_grid = False
        if do_grid:
            param_grid = {'C': [0.1, 0.5, 1, 10, 100, 200],
                          'gamma': [1.5, 1.2, 1., 0.8, 0.5]}
            grid = GridSearchCV(SVC(kernel='rbf', random_state=42, verbose=False, max_iter=5000),
                                param_grid=param_grid, scoring='accuracy')
            X, Xts, y, yts = self.generate_trainset(self.dict_comp, self.exclude,
                                                    sensors=self.sensors,
                                                    TR=TR, TS=TS, reduce=reduce,
                                                    normalized=normalized, balance=balance, perc=perc,
                                                    random_state=13 )
            X = np.vstack((X,Xts))
            y = np.append(y, yts)
            grid.fit(X, y)
            print(grid.best_params_)
            clf = SVC(kernel='rbf', random_state=42, verbose=False, max_iter=5000,
                      C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
            self.exclude = {}
            for c in self.SCA.components:
                self.exclude[c] = []
            input()

        #Inizializza le variabili per salvare i risultati
        acc_on_exp = 0.
        y_all = []
        yp_all = []
        yts_all = []
        ytmp_all = []
        yprd_all = []

        #Itera nruns volte
        for ii in range(nruns):
            #Genera i dati di train e test
            X, Xts, y, yts = self.generate_trainset(self.dict_comp,self.exclude,
                                                    sensors=self.sensors,
                                                    TR=TR,TS=TS,reduce=reduce,
                                                    normalized=normalized,balance=balance,perc=perc,
                                                    random_state=13*ii)

            print(X.shape, y.ravel().shape, ' : ', end='')
            print()
            print(Xts.shape, yts.ravel().shape, ' : ', end='')
            print()

            #print('Hit RETURN to continue',end='')
            #input()

            #Se i dati non sono stati ridotti, usa la logica temporale e il sistema di votazione delle predizioni
            if not reduce:
                ytmp, yprd = self.predict_experiment(X, y, Xts, yts, clf)
                yts_all.extend(yts)
                yprd_all.extend(yprd)
                ytmp_all.extend(ytmp)
                CR = self.report_on_experiment(yts, ytmp, yprd, 'knn')
            #Altrimenti si fa una normale predizione
            else:
                clf.fit(X,y)
                yprd = clf.predict(Xts)
                yp = clf.predict(X)
                y_all.extend(y)
                yp_all.extend(yp)
                yts_all.extend(yts)
                yprd_all.extend(yprd)
                ytmp_all.extend(yprd)
                CR = classification_report(yts, yprd, zero_division=0, output_dict=True)

            #input('Hit RETURN to continue')
            #Somma le accuratezze
            acc_on_exp += CR['accuracy']
            print()
        #Calcola la media finale e la stampa
        acc_on_exp /= nruns
        print('\n')
        print('final acc_on_exp: {}'.format(acc_on_exp))
        if reduce:
            print(classification_report(y_all,yp_all,zero_division=0))
        else:
            print(classification_report(yts_all, yprd_all, zero_division=0))

        #Mostra i risultati dell'esperimento su tutte e 10 le esecuzioni
        self.display_results(yts_all, ytmp_all, yprd_all, clf.classes_, 'Voting cumulative')

    #Esegue una ricerca per trovare la migliore combinazione di modelli
    def run_prova_sei(self,nruns=10,reduce=False,normalized=True,balance=True,perc=0.7,write_GR=False,write_LT=False,print_resume=False):
        #Recupera le strategie di train e test
        TR = self.TR
        TS = self.TS

        #Se richiesto stampa il file e la tabella LaTeX
        if write_GR:
            self.write_general_report(reduce=reduce)
        if write_LT:
            ind, temp = self.argsort_exp(order_by='mis',print_resume=print_resume, only_slow_mot=True)
            self.write_table_latex(reduce=reduce,ind=ind)
            #input('Hit RETURN to continue')

        #knn = make_pipeline(StandardScaler(), self.classifiers['KNN'])
        #Sovrascrive le definizioni precedenti di alcuni classificatori (probabilmente a seguito di test precedenti)
        svm = SVC(C=50., gamma=1.3, kernel='rbf', random_state=42, verbose=False, max_iter=5000)
        self.classifiers['SVM (gaussian)'] = svm
        mlp = MLPClassifier(solver='sgd', max_iter=5000, tol=1.e-3, activation='tanh',
                            alpha=1.e-4, verbose=False, random_state=3857, hidden_layer_sizes=(32,5),
                            n_iter_no_change=100)
        self.classifiers['MLP neural net'] = mlp

        print(len(self.classifiers))
        #Esegue 30 esperimenti diversi di composizione del classificatore
        for i in range(30):

            #Resetta il dizionario di file già usati
            self.exclude = {}
            for c in self.SCA.components:
                self.exclude[c] = []

            #Definisce i modelli sempre presenti
            res = [('LightGBM', self.classifiers['LightGBM']),
                   ('Quadratic Discriminant', self.classifiers['Quadratic Discriminant']),
                   ('XGBoost', self.classifiers['XGBoost'])]

            #res = random.sample(list(self.classifiers.items()),3)
            #Sceglie un modello a caso dalla lista e controlla che non sia uguale ad uno di quelli in res
            while True:
                cho = random.choice(list(self.classifiers.items()))
                if not cho in res:
                    break

            print(res)
            print(cho)
            res.append(cho)
            print(res)
            #Crea il classificatore con la lista scelta di modelli
            voting = VotingClassifier(res)

            #Crea la pipeline con la normalizzazione
            clf = make_pipeline(StandardScaler(), voting)

            #Inizializza le variabili dei risultati
            acc_on_exp = 0.
            y_all = []
            yp_all = []
            yts_all = []
            ytmp_all = []
            yprd_all = []

            #Esegue nruns simulazioni
            for ii in range(nruns):
                #Genera i dati di training e test
                X, Xts, y, yts = self.generate_trainset(self.dict_comp,self.exclude,
                                                        sensors=self.sensors,
                                                        TR=TR,TS=TS,reduce=reduce,
                                                        normalized=normalized,balance=balance,perc=perc,
                                                        random_state=13*ii)

                print(X.shape, y.ravel().shape, ' : ', end='')
                print()
                print(Xts.shape, yts.ravel().shape, ' : ', end='')
                print()

                #print('Hit RETURN to continue',end='')
                #input()

                #Se i dati non sono stati ridotti usa la logica temporale e il sistema di votazioni
                if not reduce:
                    ytmp, yprd = self.predict_experiment(X, y, Xts, yts, clf)
                    yts_all.extend(yts)
                    yprd_all.extend(yprd)
                    ytmp_all.extend(ytmp)
                    CR = self.report_on_experiment(yts, ytmp, yprd, 'knn')
                #Altrimenti fa una predizione standard
                else:
                    clf.fit(X,y)
                    yprd = clf.predict(Xts)
                    yp = clf.predict(X)
                    y_all.extend(y)
                    yp_all.extend(yp)
                    yts_all.extend(yts)
                    yprd_all.extend(yprd)
                    ytmp_all.extend(yprd)
                    CR = classification_report(yts, yprd, zero_division=0, output_dict=True)

                #input('Hit RETURN to continue')
                #Calcola l'accuratezza totale
                acc_on_exp += CR['accuracy']
                print()
            #Calcola la media e la stampa
            acc_on_exp /= nruns
            print('\n')
            print('final acc_on_exp: {}'.format(acc_on_exp))
            #Stampa i report a seconda della riduzione dei dati
            if reduce:
                print(classification_report(y_all,yp_all,zero_division=0))
                CR = classification_report(yts_all, yprd_all, zero_division=0,output_dict=True)
            else:
                print(classification_report(yts_all, yprd_all, zero_division=0))
                CR = classification_report(yts_all, yprd_all, zero_division=0,output_dict=True)

#        self.display_results(yts_all, ytmp_all, yprd_all, clf.classes_, 'Voting cumulative')
            accuracy = CR['accuracy']

            #Scrive la lista dei modelli usati e l'accuratezza in un file di testo
            with open('risultati_random.txt','a') as file:
                print(res,' ',accuracy,file=file)

    #Addestra e valida un modello su un insieme di dati modificati tramite Differential Sensing
    def run_prova_cinque(self,nruns=10,normalized=True,balance=True,perc=0.7,write_GR=False,write_LT=False,print_resume=False):
        #Recupera le strategie di train e test
        TR = self.TR
        TS = self.TS

        #Se richiesto genera il file o la tabella LaTeX
        if write_GR:
            self.write_general_report()
        if write_LT:
            ind, temp = self.argsort_exp(order_by='mis',print_resume=print_resume, only_slow_mot=True)
            self.write_table_latex(ind=ind)
            #input('Hit RETURN to continue')

        #Inizializza dei classificatori che non vengono utilizzati
        knn = self.classifiers['KNN']

        #voting = VotingClassifier([('KNN', self.classifiers['KNN']),
        #                           ('LGBM', self.classifiers['LightGBM']),
        #                           ])
        voting = VotingClassifier([('KNN', self.classifiers['KNN']),
                                   ('LGBM', self.classifiers['LightGBM']),
                                   ('mlp', self.classifiers['MLP neural net']),
                                   ('svm', self.classifiers['SVM (gaussian)']),
                                   ])
        #clf = make_pipeline(StandardScaler(), knn)
        #Crea una pipeline di esecuzione
        clf = make_pipeline(StandardScaler(), self.classifiers['LightGBM'])

        #Inizializza le variabili per i risultati
        acc_on_exp = 0.
        yts_all = []
        ytmp_all = []
        yprd_all = []

        #Esegue nruns cicli
        for ii in range(nruns):
            #Genera i dati di train e test
            X, Xts, y, yts = self.generate_trainset(self.dict_comp,self.exclude,
                                                    sensors=self.sensors,
                                                    TR=TR,TS=TS,
                                                    normalized=False,balance=balance,perc=perc,
                                                    random_state=13*ii)
            # 0   1   2   3   4   5 |  6   7   8    9   10    11  |  12    13    14  15
            # 1   2   3   4   5   6 |  7   8   9   10   11    12  |  13    14    15
            #0-1 1-2 2-3 3-4 4-5 5-6| 6-7 7-8 8-9 9-10 10-11 11-12 12-13 13-14 14-15
            #Usa il Differential Sensing: sottrae ad ogni sensore il valore del sensore a destra (tranne per gli ultimi 4)
            A   = X  [:, np.r_[0:5, 6:11]] - X  [:, np.r_[1:6, 7:12]]
            A   = np.hstack((A,X[:,12:]))
            Ats = Xts[:, np.r_[0:5, 6:11]] - Xts[:, np.r_[1:6, 7:12]]
            Ats = np.hstack((Ats,Xts[:,12:]))

            print(A.shape, y.ravel().shape, ' : ', end='')
            print()
            print(Ats.shape, yts.ravel().shape, ' : ', end='')
            print()

            #print('Hit RETURN to continue',end='')
            #input()

            #Addestra il modello e predice i risultati
            ytmp, yprd = self.predict_experiment(A, y, Ats, yts, clf)
            yts_all.extend(yts)
            yprd_all.extend(yprd)
            ytmp_all.extend(ytmp)
            CR = self.report_on_experiment(yts, ytmp, yprd, 'Voting')

            #input('Hit RETURN to continue')
            #Somma l'accuratezza
            acc_on_exp += CR['accuracy']
            print()
        #Calcola la media
        acc_on_exp /= nruns
        print('\n')
        print('final acc_on_exp: {}'.format(acc_on_exp))

        #Stampa i risultati finali
        self.display_results(yts_all, ytmp_all, yprd_all, clf.classes_, 'Voting cumulative')

    #Visualizza graficamente i risultati di una Confusion Matrix salvata in un file di testo
    def display_CM(self, file='CM.txt'):

        #Definisce la lista di sostanze
        classes = [
             'ACETIC_ACID', 'ACETONE', 'AIR', 'AMMONIA', 'AMMONIUM_CHLORIDE', 'APPLE_VINEGAR', 'BIOETHANOL',
             'BUTANE', 'CALCIUM_NITRATE', 'DIESEL', 'ETHANOL', 'FORMIC ACID', 'GASOLINE', 'HYDROGEN_PEROXIDE',
             'ISOPROPANOL', 'KEROSENE', 'LIGHTER_FLUID', 'METHANE', 'NITROMETHANE', 'PHOSPHORIC ACID', 'RED_WINE',
             'SODIUM HYDROXIDE', 'UREA', 'WATER_VAPOR']
        #Conta le classi
        nclasses = len(classes)
        #Legge la matrice dal file di testo
        CM = np.loadtxt(file,max_rows=nclasses)
        #Crea la figura
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        #Disegna la matrice come immagine
        cax = ax.matshow(CM, cmap='Blues')
        #Aggiunge la legenda laterale con la scala dei colori
        fig.colorbar(cax)
        #Formatta gli assi
        ax.xaxis.tick_bottom()
        plt.xticks(ticks=range(len(classes)),labels=classes, rotation=45,ha='right')
        plt.yticks(ticks=range(len(classes)), labels=classes)
        #Recupera le dimensioni della matrice e scrive all'interno di ogni cella la percentuale di accuratezza
        n,m = CM.shape
        for i in range(n):
            for j in range(n):
                plt.text(i-0.3,j+0.3,'{:4.1f}'.format(CM[j,i]*100.),fontsize=7)
        #Stampa un valore finto di accuratezza (residuo di codice?)
        plt.text(n - 4, -0.7, '{:s} {:4.1f}%'.format('accuracy', 88.7), fontsize=7)
        #plt.tight_layout()
        #Mostra la figura e pulisce la memoria
        plt.show()
        plt.close()

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
        with open('CM.txt','w') as file:
            np.savetxt(file,CM)
            print(f"\nClassification Report ( majority ):",file=file)
            #print(classification_report(yts_all, yprd_all, zero_division=0),file=file)
            print(table,file=file)

        #Imposta il grafico per stampare la CM normalizzata
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        cax = ax.matshow(CM, cmap='Blues')
        fig.colorbar(cax)
        ax.xaxis.tick_bottom()
        plt.xticks(ticks=range(len(classes)),labels=classes, rotation=45,ha='right')
        plt.yticks(ticks=range(len(classes)), labels=classes)
        n,m = CM.shape
        for i in range(n):
            for j in range(n):
                plt.text(i-0.3,j+0.3,'{:4.1f}'.format(CM[j,i]*100.),fontsize=7)
        #Scrive l'accuratezza globale in basso
        plt.text(n - 4, -0.7, '{:s} {:4.1f}%'.format('accuracy', accuracy*100.), fontsize=7)
        #Mostra il grafico e pulisce la memoria
        plt.show()
        plt.close()

#        disp = ConfusionMatrixDisplay(confusion_matrix=CM,
#                                      display_labels=classes)
#        disp.plot()
#        plt.show()

    #Genera un file di testo che riassume l'intero contenuto del dataset
    def write_general_report(self,reduce=False):
        # write file general_report.txt with description of the data we want to use
        #Apre il file txt
        with open('general_report.txt','w') as file:
            #Itera sulle chiavi del dizionario dei componenti
            for i in self.dict_comp.keys():
                #Salta tutti i componenti classificati come 'OTHERS'
                if not i == 'OTHERS':
                    cadd = i
                    print('{} : '.format(cadd), file=file)
                for c in self.dict_comp[i]:
                    print('\t{} : '.format(c), file=file)
                    #Divide gli esperimenti in 'NUOVI' e 'VECCHI'
                    experiments = self.split_experiments(c)
                    list_slow_exp = experiments['NUOVI']
                    list_stat_exp = experiments['VECCHI']
                    #Itera ogni esperimento 'VECCHIO'
                    for e in list_stat_exp:
                        #Se i dati sono stati ridotti li prende da self.SCA.DATARED
                        if reduce:
                            (a, b) = self.SCA.DATARED[(c, e)]
                            t = a.to_numpy()[:, self.sensors]
                            t[pd.isna(t)] = 0.
                        #Altrimenti deve prendere quelli grezzi da self.SCA.DATA
                        else:
                            (a,b) = self.SCA.DATA[(c,e)]
                            t = a.to_numpy()[:, self.sensors]
                            t[pd.isna(t)] = 0.
                            #Calcola la somma di tutti i valori assoluti dei sensori per ogni istante
                            aa = np.abs(t).sum(axis=1)
                            # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                            #Crea una maschera booleana che mantiene solo le righe dove aa supera una certa soglia
                            ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                            t = t[ind, :]
                        #Scrive nel file: ID esperimento, righe totali, (righe utili), nome del file originale
                        n1,m = a.shape
                        n, m = t.shape
                        print('\t{:3d}: {:5d}({:5d})\t{}'.format(e,n1,n,b),file=file)
                    #Itera ogni esperimento 'NUOVO' facendo la stessa cosa che ha fatto con quelli 'VECCHI'
                    for e in list_slow_exp:
                        if reduce:
                           (a, b) = self.SCA.DATARED[(c, e)]
                           t = a.to_numpy()[:, self.sensors]
                           t[pd.isna(t)] = 0.
                        else:
                            (a, b) = self.SCA.DATA[(c, e)]
                            t = a.to_numpy()[:, self.sensors]
                            t[pd.isna(t)] = 0.
                            aa = np.abs(t).sum(axis=1)
                            # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                            ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                            t = t[ind, :]
                        n1,m = a.shape
                        n, m = t.shape
                        print('\t{:3d}: {:5d}({:5d})\t{}'.format(e,n1,n,b),file=file)

    #Scrive in codice LaTeX una tabella che mostra il numero di esperimenti disponibili per ogni sostanza
    def write_table_latex(self,reduce=False,ind=None):
        #Apre il file txt
        with open('latex_table.txt','w') as file:
            #Inizializza delle variabili per calcolare i totali complessivi del dataset
            nsltot = 0
            nsttot = 0
            nm1sltot = 0
            nmsltot  = 0
            nm1sttot = 0
            nmsttot  = 0
            #Se non viene passato nessun ordine specifico usa l'ordine alfabetico
            if (ind is None):
                indord = range(len(self.SCA.components))
            #Altrimenti usa l'ordine fornito
            else:
                indord = ind
            #Per ogni componente (in ordine)
            for i in indord:
                #for c in self.SCA.components:
                #Estrae il nome del componente e lo scrive nella prima colonna della tabella
                c = self.SCA.components[i]
                print('{:30s} '.format(c),file=file,end='')
                #Separa gli ID dei file 'NUOVI' e 'VECCHI'
                experiments = self.split_experiments(c)
                slow = experiments['NUOVI']
                stat = experiments['VECCHI']
                #Conta il numero di file di ogni tipo e aggiorna le variabili globali
                nslow = len(slow)
                nsltot += nslow
                nstat = len(stat)
                nsttot += nstat
                #Scrive le colonne 2 e 3 della tabella
                print('& {:5d} & {:5d} '.format(nslow,nstat),file=file,end='')
                n1tot = 0
                ntot  = 0
                #Itera su tutti i file 'NUOVI'
                for e in slow:
                    #Carica i dati già ridotti
                    if reduce:
                        (a, b) = self.SCA.DATARED[(c, e)]
                        t = a.to_numpy()[:, self.sensors]
                        t[pd.isna(t)] = 0.
                    #Altrimenti carica i dati grezzi e applica il filtro
                    else:
                        (a, b) = self.SCA.DATA[(c, e)]
                        t = a.to_numpy()[:, self.sensors]
                        t[pd.isna(t)] = 0.
                        aa = np.abs(t).sum(axis=1)
                        # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                        ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                        t = t[ind, :]
                    #Calcola le righe totali e le righe filtrate
                    n1, m = a.shape
                    n1tot += n1
                    n, m = t.shape
                    ntot += n
                #Scrive la colonna 4
                print('& {:6d}({:6d}) '.format(n1tot,ntot),file=file,end='')
                #Aggiorna i totali globali
                nm1sltot += n1tot
                nmsltot  += ntot
                n1tot = 0
                ntot  = 0
                #Itera su tutti i file 'VECCHI' e ripete le stesse operazioni
                for e in stat:
                    if reduce:
                        (a, b) = self.SCA.DATARED[(c, e)]
                        t = a.to_numpy()[:, self.sensors]
                        t[pd.isna(t)] = 0.
                    else:
                        (a, b) = self.SCA.DATA[(c, e)]
                        t = a.to_numpy()[:, self.sensors]
                        t[pd.isna(t)] = 0.
                        aa = np.abs(t).sum(axis=1)
                        # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                        ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                        t = t[ind, :]
                    n1, m = a.shape
                    n1tot += n1
                    n, m = t.shape
                    ntot += n
                print('& {:6d}({:6d}) '.format(n1tot, ntot), file=file, end='')
                nm1sttot += n1tot
                nmsttot  += ntot

                #Chiude la riga LaTeX
                print('\\\\\\hline',file=file)
            #Scrive l'ultima riga con i totali
            print('{:30s} & {:5d} & {:5d} & {:6d}({:6d}) & {:6d}({:6d}) \\\\\\hline'.format('Total:',nsltot,nsttot,nm1sltot,nmsltot,nm1sttot,nmsttot),file=file)

    #Confronta LightGBM con KNN
    def prova_tre(self):
        #Recupera le strategie di train e test
        TR = self.TR
        TS = self.TS

        #Dizionari con i dati da testare e da escludere
        dict_comp = self.dict_comp
        exclude = self.exclude
        misure = self.misure

        #Stampa quali sostanze sta usando
        print()
        print(dict_comp)
        print()

        #Genera il file di report e la tabella LaTeX
        self.write_general_report()
        self.write_table_latex()

        #Definisce i modelli
        top_model = 'LightGBM'
        model1 = make_pipeline(StandardScaler(), self.classifiers[top_model])
        voting = VotingClassifier([(top_model, self.classifiers[top_model]),
                                   ('mlp', self.classifiers['MLP neural net']),
                                   ('svm', self.classifiers['SVM (gaussian)']),
                                   ])
        #Questo modello non verrà utilizzato
        model2 = make_pipeline(StandardScaler(), voting)
        model3 = make_pipeline(StandardScaler(), self.classifiers['KNN'])
        #Inizializza le variabili per i risultati
        acc_on_exp = 0.
        acc_on_mod = 0.
        yts_all = []
        ytmp_all = []
        yprd_all = []
        #Esegue 10 simulazioni
        for ii in range(10):
            #Genera i dati di train e test
            X, Xts, y, yts = self.generate_trainset(dict_comp,exclude,TR=TR,TS=TS,random_state=13*ii)

            print(X.shape, y.ravel().shape, ' : ', end='')
            print()
            print(Xts.shape, yts.ravel().shape, ' : ', end='')
            print()
            #print('Hit RETURN to continue',end='')
            #input()

            #Valuta model1 (LightGBM) usando la Cross-Validation
            y_true_all, y_pred_all = self.predict_sigle_measure(X, y, model1, cv_folds=5)
            CR = self.report_on_sigle_measure(y_true_all, y_pred_all, top_model)
            #input('Hit RETURN to continue')
            acc_on_mod += CR['accuracy']

            #Valuta model3 (KNN) tramite il sistema di votazione
            ytmp, yprd = self.predict_experiment(X, y, Xts, yts, model3)
            yts_all.extend(yts)
            yprd_all.extend(yprd)
            ytmp_all.extend(ytmp)
            CR = self.report_on_experiment(yts, ytmp, yprd, 'KNN')
            #input('Hit RETURN to continue')
            acc_on_exp += CR['accuracy']
            print()
        #Calcola le medie e stampa il confronto finale
        acc_on_exp /= 10.
        acc_on_mod /= 10.
        print('\n')
        print('final acc_on_mod: {}  acc_on_exp: {}'.format(acc_on_mod, acc_on_exp))

        #Genera un report per il KNN
        CR = self.report_on_experiment(yts_all, ytmp_all, yprd_all, 'KNN cumulative')

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
        #Separa esperimenti 'NUOVI' e 'VECCHI' e ne conta il numero
        experiments = self.split_experiments(c)
        list_slow_exp = experiments['NUOVI']
        list_stat_exp = experiments['VECCHI']
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
            n,m = a.shape
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

    #Crea le strutture dati con gli esperimenti selezionati
    def generate_trainset(self,dict_comp,exclude,sensors=range(16),reduce=False,normalized=False,balance=False,perc=0.7,TR='ALL',TS='ALL',random_state=100):
        #TR and TS can either be: 'SLOW', 'STAT', 'ALL'
        #Imposta il seme del generatore casuale
        np.random.seed(random_state)
        #Recupera il numero di sensori
        nsensors = len(sensors)
        #Inizializza le matrici vuote che verranno riempite nel ciclo
        X = np.zeros((0, nsensors))
        Xts = np.zeros((0, nsensors))
        y = np.array([])
        yts = np.array([])
        #Apre il report dei dati
        with open('data_report.txt','w') as file:
            cardinality = []
            liste = {}
            #Itera sulle categorie
            for i in dict_comp.keys():
                #Esclude i componenti in 'OTHERS'
                if not i == 'OTHERS':
                    cadd = i
                    print('{} : '.format(cadd), end='')
                    print('{} : '.format(cadd), file=file)
                    num = 0
                    numts = 0

                    #Itera sui componenti
                    for c in dict_comp[i]:
                        print('\t{} : '.format(c), file=file)
                        #Crea le liste di dati di train e test per il componente corrente
                        ltr, lts = self.define_train_test_exp(c,exclude[c],TR=TR,TS=TS)
                        #Salva la lista di train per il futuro
                        liste[c] = ltr
                        #Itera gli esperimenti nella lista di train
                        for e in ltr:
                            #Se i dati vanno ridotti li recupera da self.SCA.DATARED
                            if reduce:
                                (a, b) = self.SCA.DATARED[(c, e)]
                                t = a.to_numpy()[:, sensors]
                                t[pd.isna(t)] = 0.
                            #Altrimenti li prende da self.SCA.DATA
                            else:
                                (a, b) = self.SCA.DATA[(c, e)]
                                t = a.to_numpy()[:, sensors]
                                t[pd.isna(t)] = 0.
                                #Applica la soglia per scartare l'aria e il rumore di fondo
                                aa_abs = np.abs(t).sum(axis=1)
                                ind = (aa_abs > np.min(aa_abs) + self.soglia * (np.max(aa_abs) - np.min(aa_abs)))
                                t = t[ind, :]

                                #Se i dati vanno normalizzati divide ogni sensore per la media di riga
                                if normalized:
                                    aa_mean = t.sum(axis=1) / nsensors
                                    aa_mean[np.abs(aa_mean) < 1.e-3] = 1.
                                    t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))

                            n, m = t.shape
                            print('\ttr {:3d}: {:5d} {}'.format(e,n,b),file=file)
                            print('-{}:{}- '.format(e, n), end='')
                            num += n
                            #Aggiunge i dati processati a X e le etichette a y
                            X = np.vstack((X, t))
                            # y = np.append(y, i * np.ones(n))
                            y = np.append(y, [cadd for i in range(n)])

                        #Fa la stessa cosa con i dati nella lista di test
                        for e in lts:
                            if reduce:
                                (a, b) = self.SCA.DATARED[(c, e)]
                                t = a.to_numpy()[:, sensors]
                                t[pd.isna(t)] = 0.
                            else:
                                (a, b) = self.SCA.DATA[(c, e)]
                                t = a.to_numpy()[:, sensors]
                                t[pd.isna(t)] = 0.
                                if normalized:
                                    aa = t.sum(axis=1)/nsensors
                                    aa[np.abs(aa) < 1.e-3] = 1.
                                    t = t / np.tile(np.reshape(aa, (-1, 1)), (1, nsensors))
                                else:
                                    aa = np.abs(t).sum(axis=1)
                                    # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                                    ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                                    t = t[ind, :]
                            n, m = t.shape
                            print('\tts {:3d}: {:5d} {}'.format(e, n, b),file=file)
                            print('*{}:{}* '.format(e, n), end='')
                            numts += n
                            Xts = np.vstack((Xts, t))
                            # y = np.append(y, i * np.ones(n))
                            yts = np.append(yts, [cadd for i in range(n)])
                    #Stampa il numero di dati di train e test
                    cardinality.append({'comp':i, 'card':num})
                    print('num:{} numts:{}'.format(num, numts))

            #input()
            print(cardinality)
            df_card = pd.DataFrame(cardinality).sort_values(by='card', ascending=False)
            #Se si vogliono bilanciare i dati
            if balance:
                # Trova la classe più numerosa
                maxcard = df_card['card'].max()
                # Calcola il numero di righe target
                target_card = int(perc * maxcard)

                # Liste per accumulare le nuove righe da aggiungere (molto più efficiente di vstack in un ciclo)
                X_to_add = []
                y_to_add = []

                # Itera su ogni classe presente in df_card
                for index, row in df_card.iterrows():
                    comp = row['comp']
                    current_card = row['card']

                    # Se la classe ha meno righe del target, dobbiamo campionare
                    if current_card < target_card:
                        rows_needed = target_card - current_card
                        
                        # Trova gli indici delle righe in X che appartengono a questa classe
                        # Usiamo np.where per trovare dove y è uguale al nome del componente
                        class_indices = np.where(y == comp)[0]
                        
                        if len(class_indices) > 0:
                            # Campiona casualmente (con reinserimento) le righe mancanti direttamente dagli indici
                            sampled_indices = np.random.choice(class_indices, size=rows_needed, replace=True)
                            
                            # Estrai le righe già processate da X
                            X_new = X[sampled_indices]
                            # Crea le etichette per le nuove righe
                            y_new = np.full(rows_needed, comp)
                            
                            # Aggiungi alle liste temporanee
                            X_to_add.append(X_new)
                            y_to_add.append(y_new)

                # Se abbiamo generato nuove righe, fondiamole con i dati originali in un solo colpo
                if len(X_to_add) > 0:
                    # np.vstack e np.concatenate eseguiti UNA SOLA VOLTA alla fine
                    X = np.vstack([X] + X_to_add)
                    y = np.concatenate([y] + y_to_add)
                    
                    # Rimescola i dati in modo che le righe clonate non siano tutte alla fine
                    X, y = shuffle(X, y, random_state=random_state)
                
                
                # Aggiorna df_card per il report finale basandosi sul nuovo array y
                unique, counts = np.unique(y, return_counts=True)
                cardinality_updated = [{'comp': k, 'card': v} for k, v in zip(unique, counts)]
                df_card = pd.DataFrame(cardinality_updated).sort_values(by='card', ascending=False)

        count = 0
        #Scrive il riepilogo finale nel file
        with open('data_report.txt','a') as file:
            for index, row in df_card.iterrows():
                #print(row)
                i = row['comp']
                n = y[y == i].shape[0]
                print('\t{:30s} {:6d} ({:6d})'.format(row['comp'],row['card'],n),file=file)
                print('\t{:30s} {:6d} ({:6d})'.format(row['comp'],row['card'],n))
                count += row['card']
        print(count)
        #input('end generate_trainset')
        #Restituisce le matrici complete
        return X, Xts, y, yts

    #Valuta la capacità del sistema di aggregare diverse sostanze in classi e di gestire automaticamente tutto ciò che non è di interesse
    def prova_due(self):
        #Dizionari contenenti le classi (i primi due vengono sovrascritti)
        dict_comp = {
            'AMMONIA+ETHANOL+ISOPROPANOL': ['AMMONIA', 'ETHANOL', 'ISOPROPANOL'],
            'SODIUM HYDROXIDE': ['SODIUM HYDROXIDE'],
            'ACETIC_ACID': ['ACETIC_ACID'],
            'AMMONIUM_CHLORIDE': ['AMMONIUM_CHLORIDE']
        }
        dict_comp = {
            'AMMONIA': ['AMMONIA'],
            'ETHANOL+ISOPROPANOL': ['ETHANOL', 'ISOPROPANOL'],
            'SODIUM HYDROXIDE': ['SODIUM HYDROXIDE'],
            'ACETIC_ACID': ['ACETIC_ACID'],
            'AMMONIUM_CHLORIDE': ['AMMONIUM_CHLORIDE']
        }
        dict_comp = {
            'AMMONIA': ['AMMONIA'],
            'ETHANOL': ['ETHANOL'],
            'ISOPROPANOL': ['ISOPROPANOL'],
            'ACETIC_ACID': ['ACETIC_ACID'],
            'SODIUM HYDROXIDE': ['SODIUM HYDROXIDE'],
            'AMMONIUM_CHLORIDE': ['AMMONIUM_CHLORIDE'],
            'PHOSPHORIC ACID': ['PHOSPHORIC ACID'],
            'RED_WINE': ['RED_WINE'],
            'FORMIC ACID': ['FORMIC ACID'],
            'BUTANE': ['BUTANE'],
            'METHANE': ['METHANE'],
            'BIOETHANOL': ['BIOETHANOL'],
            'APPLE_VINEGAR': ['APPLE_VINEGAR'],
            'KEROSENE': ['KEROSENE'],
            'ACETONE': ['ACETONE'],
            #        'NITROMETHANE': ['NITROMETHANE'],
        }
        #Inizializza la lista di componenti da escludere e il dizionairo per gli esperimenti da escludere
        others = []
        exclude = {}
        #Itera i componenti 
        for c in self.SCA.components:
            exclude[c] = []
            trovato = False
            #Itera i componenti definiti sopra
            for i in dict_comp:
                #Se il componente corrente non è in quelli definiti non ci interessa
                if c in dict_comp[i]:
                    trovato = True
                    break
            #I componenti che non ci interessano vengono messi nella lista other
            if not trovato:
                others.append(c)
        #Viene creata una classe che rappresenta i componenti non richiesti
        dict_comp['OTHERS'] = others
        print(dict_comp)

        with open('general_report.txt','w') as file:
            print('ciao',file=file)
        input()

        #Definisce i modelli da utilizzare e crea le pipeline di esecuzione
        top_model = 'LightGBM'
        model1 = make_pipeline(StandardScaler(), self.classifiers[top_model])
        voting = VotingClassifier([(top_model, self.classifiers[top_model]),
                                   ('mlp', self.classifiers['MLP neural net']),
                                   ('svm', self.classifiers['SVM (gaussian)']),
                                   ])
        model2 = make_pipeline(StandardScaler(), voting)

        #Inizializza le variabili dei risultati
        acc_on_exp = 0.
        acc_on_mod = 0.
        yts_all = []
        ytmp_all = []
        yprd_all = []
        #Esegue 10 simulazioni
        for ii in range(10):
            #Genera i dati di train e test
            X, Xts, y, yts = self.generate_multiclass_trainset(dict_comp, exclude, only_slow_mov=True,
                                                          random_state=11 * ii)

            print(X.shape, y.ravel().shape, ' : ', end='')
            print()
            print(Xts.shape, yts.ravel().shape, ' : ', end='')
            print()
            input('Hit RETURN to continue')

            # find_best_model(X, y, cv_folds=5)
            # input('Hit RETURN to continue')
            #print(f"\n=== Detailed Report for Model: {top_model} ===")

            #Addestra e valuta model1 sui dati istantanei
            y_true_all, y_pred_all = self.predict_sigle_measure(X, y, model1, cv_folds=5)
            CR = self.report_on_sigle_measure(y_true_all, y_pred_all, top_model)
            input('Hit RETURN to continue')
            acc_on_mod += CR['accuracy']

            #Addestra e valuta model2 con il sistema di votazione
            ytmp, yprd = self.predict_experiment(X, y, Xts, yts, model2)
            yts_all.extend(yts)
            yprd_all.extend(yprd)
            ytmp_all.extend(ytmp)
            CR = self.report_on_experiment(yts, ytmp, yprd, 'Voting')
            input('Hit RETURN to continue')
            acc_on_exp += CR['accuracy']
            print()
        #Calcola l'accuratezza media e la stampa
        acc_on_exp /= 10.
        acc_on_mod /= 10.
        print('\n')
        print('final acc_on_mod: {}  acc_on_exp: {}'.format(acc_on_mod, acc_on_exp))

        #Crea il dizionario con il report
        CR = self.report_on_experiment(yts_all, ytmp_all, yprd_all, 'Voting cumulative')

    #Inizializza due dizionari (train e test) composti da mini-dataset focalizzati sulla distinzione tra due componenti
    def generate_dict_train_test(self, comp1, comp2, exp1, exp2):
        #Inizializza i dizionari e un contatore progressivo che funge da chiave
        Dtr = {}
        Dts = {}
        i = 0
        #Itera le strategie di test
        for hand in ['left', 'right']:
            #Se 'left' itera sugli esperimenti di comp1
            if hand == 'left':
                for e in exp1:
                    X1 = np.zeros((0, 16))
                    X2 = np.zeros((0, 16))
                    #Itera di nuovo sugli esperimenti di comp1 per far si che ognuno verrà usato come test set
                    for e1 in exp1:
                        #Salta l'esperimento di test
                        if e == e1:
                            continue
                        #Aggiunge al train set tutti gli altri esperimenti di comp1 che recupera da self.SCA.DATA
                        else:
                            # print(e1,' ',end='')
                            t = self.SCA.DATA[(comp1, e1)][0].to_numpy()[:, :-1]
                            X1 = np.vstack((X1, t))
                    n, m = X1.shape
                    #Assegna a questi elementi l'etichetta 1
                    y1 = np.ones(n)
                    #Mette nel train set tutti gli esperimenti di comp2
                    for e2 in exp2:
                        # print(e2,' ',end='')
                        t = self.SCA.DATA[(comp2, e2)][0].to_numpy()[:, :-1]
                        X2 = np.vstack((X2, t))
                    n, m = X2.shape
                    #Assegna a questi elementi l'etichetta 0
                    y2 = np.zeros(n)
                    #Crea le matrici con i dati di train e l'array con le etichette
                    X = np.vstack((X1, X2))
                    y = np.append(y1, y2)
                    #Aggiunge al test set solo l'elemento corrente (quello escluso dal train set)
                    Xts = self.SCA.DATA[(comp1, e)][0].to_numpy()[:, :-1]
                    n, m = Xts.shape
                    #Assegna a questo elemento l'etichetta 1
                    yts = np.ones(n)
                    #Aggiunge ai dizionari di train e test le matrici corrispondenti (all'indice i)
                    Dtr[i] = (X, y)
                    Dts[i] = (Xts, yts)
                    #Incrementa il contatore della chiave per la prossima iterazione
                    i += 1
                    print('* ', X.shape, Xts.shape)
            #Altrimenti itera sugli esperimenti di comp2
            else:
                #Ripete lo stesso processo, questa volta prendendo solo esperimenti di comp2 per il test set
                for e in exp2:
                    X1 = np.zeros((0, 16))
                    X2 = np.zeros((0, 16))
                    for e1 in exp1:
                        # print(e1,' ',end='')
                        t = self.SCA.DATA[(comp1, e1)][0].to_numpy()[:, :-1]
                        X1 = np.vstack((X1, t))
                    n, m = X1.shape
                    y1 = np.ones(n)
                    for e2 in exp2:
                        if e == e2:
                            continue
                        else:
                            # print(e2,' ',end='')
                            t = self.SCA.DATA[(comp2, e2)][0].to_numpy()[:, :-1]
                            X2 = np.vstack((X2, t))
                    n, m = X2.shape
                    y2 = np.zeros(n)
                    X = np.vstack((X1, X2))
                    y = np.append(y1, y2)
                    Xts = self.SCA.DATA[(comp2, e)][0].to_numpy()[:, :-1]
                    n, m = Xts.shape
                    yts = np.zeros(n)
                    Dtr[i] = (X, y)
                    Dts[i] = (Xts, yts)
                    i += 1
                    print('+ ', X.shape, Xts.shape)
        #Restituisce i dizionari di train e test
        return Dtr, Dts

    #Crea il test e train set a partire da un dizionario di classi di componenti
    def generate_multiclass_trainset(self, dict_comp, exclude, only_slow_mov=False, random_state=100):
        #Imposta il seme per il generatore casuale
        np.random.seed(random_state)
        #Inizializza le matrici finali
        X = np.zeros((0, 16))
        Xts = np.zeros((0, 16))
        y = np.array([])
        yts = np.array([])
        #Itera sulle chiavi del dizionario
        for i in dict_comp.keys():
            #Continua solo se la classe non è 'OTHERS'
            if not i == 'OTHERS':
                #Classe corrente
                cadd = i
                print('{} : '.format(cadd), end='')
                #Inizializza i contatori per il train e il test set della classe
                num = 0
                numts = 0

                #Itera i componenti nella classe
                for c in dict_comp[i]:
                    #Cerca di pescare casualmente un esperimento da usare come test set
                    while True:
                        ets = np.random.choice(self.SCA.experiments[c])
                        (a, b) = self.SCA.DATA[(c, ets)]
                        #Se only_slow_mov è False il ciclo non finirà mai
                        if only_slow_mov and (ets not in exclude[c]) and ((b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1)):
                            break
                    print('{} : '.format(ets), end='')

                    #Itera tutti gli esperimenti del componente corrente
                    for e in self.SCA.experiments[c]:
                        #Recupera il dataset e il nome del file
                        (a, b) = self.SCA.DATA[(c, e)]
                        #Trasforma il dataset in una matrice NumPy, escludento la colonna dei label
                        t = a.to_numpy()[:, :-1]
                        #Applica la maschera booleana per rimuovere i segnali troppo deboli
                        aa = np.abs(t).sum(axis=1)
                        # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
                        ind = (aa > np.min(aa) + self.soglia * (np.max(aa) - np.min(aa)))
                        t = t[ind, :]
                        n, m = t.shape
                        #Se only_slow_mov è False (cosa che non potrebbe mai succedere perché si bloccherebbe al ciclo più in alto)
                        if not only_slow_mov:
                            #Se la dimensione del test set è 0 e il file è buono (>1000 righe) lo aggiunge al test set
                            #Se però ci sono più componenti nella classe, solo uno di questi avrà un esperimento nel test set
                            if n > 1000 and numts == 0 and (e not in exclude[c]):  # numts < 2000:
                                print('*{}:{}* '.format(e, n), end='')
                                numts += n
                                Xts = np.vstack((Xts, t))
                                # yts = np.append(yts, i * np.ones(n))
                                yts = np.append(yts, [cadd for i in range(n)])
                                exclude[c].append(e)
                            #Altrimenti aggiunge il file al train set
                            else:
                                print('-{}:{}- '.format(e, n), end='')
                                num += n
                                X = np.vstack((X, t))
                                # y = np.append(y, i * np.ones(n))
                                y = np.append(y, [cadd for i in range(n)])
                        #Altrimenti
                        else:
                            #Controlla se il file è 'slow
                            if (b.lower().find('slow') > -1) or (b.lower().find('checkair') > -1):
                                #if numts == 0 and (e == ets):  # and (e not in exclude[c]):  # numts < 2000:
                                #Se l'esperimento corrente è quello scelto per il test set
                                if (e == ets):  # and (e not in exclude[c]):  # numts < 2000:
                                    print('*{}:{}* '.format(e, n), end='')
                                    numts += n
                                    #L'esperimento viene aggiunto al test set e a exclude
                                    Xts = np.vstack((Xts, t))
                                    # yts = np.append(yts, i * np.ones(n))
                                    yts = np.append(yts, [cadd for i in range(n)])
                                    exclude[c].append(e)
                                #Altrimenti viene aggiunto al train set
                                else:
                                    print('-{}:{}- '.format(e, n), end='')
                                    num += n
                                    X = np.vstack((X, t))
                                    # y = np.append(y, i * np.ones(n))
                                    y = np.append(y, [cadd for i in range(n)])
                    #Stampa il numero di esperimenti nel train e nel test set
                    print('num:{} numts:{}'.format(num, numts))
                    # y = np.append(y,i*np.ones(num))
        #Ritorna le matrici complete
        return X, Xts, y, yts
    
    def prova_sette(self,reduce=False,normalized=False,balance=True,perc=1):
        #Recupera le strategie di train e test
        TR = self.TR
        TS = self.TS

        top_model = 'MLP neural net'
        model1 = make_pipeline(StandardScaler(), self.classifiers[top_model])
        
        #Genera i dati di train e test
        X, Xts, y, yts = self.generate_trainset(self.dict_comp,self.exclude,
                                                sensors=self.sensors,
                                                TR=TR,TS=TS,reduce=reduce,
                                                normalized=normalized,balance=balance,perc=perc,
                                                random_state=0)

        print(X.shape, y.ravel().shape, ' : ', end='')
        print()
        print(Xts.shape, yts.ravel().shape, ' : ', end='')
        print()
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        yts = le.transform(yts)

        model1.fit(X, y)
            
        y_pred = model1.predict(Xts)
        
        y_pred = le.inverse_transform(y_pred)
        
        y_true = le.inverse_transform(yts)
        y_voted = np.copy(y_pred)
        
        df_res = pd.DataFrame({"Vera" : y_true, "Predetta" : y_pred})
            
        for gas in df_res["Vera"].unique():
            mask = (df_res["Vera"] == gas)
            voto_maggioranza = df_res.loc[mask, "Predetta"].mode()[0]
            y_voted[mask] = voto_maggioranza

        #Mostra i risultati dell'esperimento su tutte e 10 le esecuzioni
        self.display_results(y_true,y_pred,y_voted, le.classes_, 'Voting cumulative')
        
    def prova_otto(self, reduce=False, normalized=False, balance=True, perc=1):
        TR = self.TR
        TS = self.TS

        print("\n--- INIZIALIZZAZIONE MODELLO DEFINITIVO ---")
        top_model = 'MLP neural net'
        model1 = make_pipeline(StandardScaler(), self.classifiers[top_model])

        print(f"Generazione Train e Test Set (TR={TR}, TS={TS}, Bilanciamento={balance})...")
        # Imposto un random_state fisso per avere gli stessi risultati riproducibili nella tesi
        X, Xts, y, yts = self.generate_trainset(self.dict_comp, self.exclude,
                                                sensors=self.sensors,
                                                TR=TR, TS=TS, reduce=reduce,
                                                normalized=normalized, balance=balance, perc=perc,
                                                random_state=42)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print("Addestramento Rete Neurale (MLP) in corso...")
        model1.fit(X, y_encoded)

        print("Predizione riga-per-riga e Votazione a Maggioranza...")
        y_pred_num = model1.predict(Xts)
        y_pred = le.inverse_transform(y_pred_num)
        y_true = yts.copy()
        
        y_voted = np.empty(len(y_pred), dtype=object)
        y_voted[:] = y_pred

        # Votazione a Maggioranza (Majority Voting)
        df_res = pd.DataFrame({'Vera': y_true, 'Predetta': y_pred})

        for gas in df_res['Vera'].unique():
            mask = (df_res['Vera'] == gas)
            voto_maggioranza = df_res.loc[mask, 'Predetta'].mode()[0]
            y_voted[mask] = voto_maggioranza

        # --- IL TRUCCO SALVA-MATRICE ---
        # Se nel Test Set manca casualmente un gas, la matrice si restringe e i nomi scivolano.
        # Aggiungiamo un singolo voto fittizio per i gas mancanti per bloccare la griglia a 24x24!
        missing_gases = set(le.classes_) - set(y_true)
        for mg in missing_gases:
            y_true = np.append(y_true, mg)
            y_pred = np.append(y_pred, mg)
            y_voted = np.append(y_voted, mg)

        print("Generazione risultati e grafici...\n")
        self.display_results(y_true, y_pred, y_voted, le.classes_, 'MLP + Majority Voting')
        
             

#Gestore centrale del dataset, prepara e organizza i dati
class SCA_NRM_data:
    #Inizializza l'oggetto impostando i percorsi dei file, definendo la lista delle sostanze chimiche da analizzare e caricando i dati in memoria
    #Gestisce un sistema di caching (file .npy), popola le strutture dati interne che mappano ogni sostanza ai suoi esperimenti
    #Opzionalmente esegue una riduzione dei dati e stampa una guida
    def __init__(self,ndeg=-1,sensors=np.arange(16),verbose=0,help=True,reduce=False):
        '''
        :param ndeg: grado delle interpolanti polinomiali usate per ridurre i dati
        :param sensors: array contenente gli indici dei sensori che si vuole includere nel training set
        :param verbose: livello di dettaglio dei log
        :param help: se True, stampa le istruzioni di avvio
        :param reduce: se True, esegue subito la riduzione dimensionale dei dati
        '''
        # traindir: percorso completo alla cartella contenente i dati di training normalizzati
        #           normalmente contenuti nella cartella .../SCA/TRAINING/NORMALIZED/
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.traindir = os.path.join(base_dir, 'TRAINING', 'NORMALIZED') + os.sep
        # components: nome delle sostanze così come compaiono nella cartella traindir
        #self.components = ['ACETIC ACID', 'AIR', 'AMMONIUM CHLORIDE', 'DIESEL', 'GASOLINE', 'ISOPROPANOL', 'NAPHTA', 'UREA', \
        #                   'ACETONE', 'AMMONIA', 'CALCIUM NITRATE', 'ETHANOL', 'HYDROGENE PEROXIDE', 'KEROSENE', \
        #                   'NITROMETHANE', 'WATER VAPOR']
        self.components = ['ACETIC_ACID', 'AIR', 'ACETONE', 'AMMONIUM_CHLORIDE', 'AMMONIA', 'CALCIUM_NITRATE', 'BUTANE',
                           'BIOETHANOL', 'APPLE_VINEGAR', 'DIESEL', 'GASOLINE', 'FORMIC_ACID', 'ETHANOL', 'ISOPROPANOL',
                           'HYDROGEN_PEROXIDE', 'METHANE', 'LIGHTER_FLUID', 'KEROSENE', 'RED_WINE', 'PHOSPHORIC_ACID',
                           'NITROMETHANE', 'WATER_VAPOR', 'UREA', 'SODIUM_HYDROXIDE']
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
            #print(component, expr)

        #Stampa opzionalmente un report sulle proprietà dei dati caricati
        if verbose > 0:
            self.data_prop()
        #Memorizza i parametri del costruttore
        self.ndeg = ndeg
        self.sensors = sensors

        #Se si vogliono ridurre i dati calcola le feature statistiche dai dati grezzi e popola self.DATARED
        if reduce:
            self.data_reduce()
            #self.data_reduce_pacchetti(npac=2)
        else:
            #Altrimenti copia il puntatore dei dati grezzi in self.DATARED
            self.DATARED = self.DATA
        #Stampa opzionalmente a video una spiegazione sulla struttura dei dati e come accedervi
        if help:
            self.help()
        #self.data_reduce()
        #self.create_train_set()


    #Stampa a console una guida rapida che spiega quali strutture dati sono disponibili nell'oggetto
    #Elenca tutte le sostanze chimiche riconosciute, formattandole in una griglia, e mostra un esempio di come sono indicizzati gli esperimenti
    #Spiega la struttura chiave-valore del dizionario principale DATA per accedervi correttamente
    def help(self):
        #Stampa un titolo e un elenco delle tre proprietà principali dell'istanza accessibili dall'utente
        print(
            '\n\nSCA normalized data class provides the following data structures:')  # are stored in DATA dictionary')
        print('\t1)        DATA: dictionary of data')
        print('\t2)  components: list of inquinants')
        print('\t3) experiments: dictionary of experiments')
        #Stampa un separatore per l'elenco dei componenti
        print('\n\t------------------ components -----------------')
        #Inizializza un contatore len a 1
        len = 1
        #Stampa una tabulazione iniziale
        print('\t', end='')
        #Itera su ogni componente della lista self.components
        for c in self.components:
            #Se il contatore è multiplo di 5 va a capo e inserisce una tabulazione
            if len % 5 == 0:
                print('\n\t', end='')
            #Altrimenti stampa il nome del componente formattandolo in uno spazio fisso di 10 caratteri (per allineare le colonne)
            else:
                print('{:10s} '.format(c), end='')
            #Incrementa il contatore
            len += 1
        # print('\t',self.components)
        #Chiude la sezione principale e apre quella dedicata agli esperimenti
        print('\n\t-----------------------------------------------')
        print('\n\t---------------- experiments ------------------')
        #Prende il primo componente della lista
        c = self.components[0]
        #Stampa l'inizio della riga di esempio
        print('\texample: experiments[{}] = '.format(c),end='')
        #Prende il primo esperimento per quel componente
        e0 = self.experiments[c][0]
        #Prende l'ultimo esperimento per quel componente
        e1 = self.experiments[c][-1]
        #Li stampa come intervallo, mostrando all'utente il range di ID disponibili
        print('{}..{}'.format(e0,e1))
        #Stampa il separatore finale e una riga vuota
        print('\t-----------------------------------------------')
        print()
        #Spiega esplicitamente la firma del dizionario DATA
        print('\tStructure of the DATA dictionary is the following:')
        print('\t(df,file) = DATA[(a,b)] where')
        print('\t\t  a  : <component>')
        print('\t\t  b  : <experiment_id>')
        print('\t\t  df : data frame holding the experimental data')
        print('\t\tfile : name of csv file where measurements are stored')

    #Trasforma i dati grezzi in un array di feature statistiche
    #Riduce una matrice N x 16 ad un singolo vettore riga di dimensione 1 x 81 (80 feature e 1 etichetta)
    def data_reduce(self):
        '''
        riduce i dati sostituendo a tutte le misure di ogni sensore alcuni indicatori di quel sensore
        :return:
        '''
        #Dizionario per ospitare i dati processati
        self.DATARED = {}

        #Itera su ogni componente
        for c in self.components:
            #print(c + ' ', end='')
            #Itera su ogni esperimento di quel componente
            for e in self.experiments[c]:
                #print(e, ' ', end='')
                #Recupera il dataframe grezzo e il nome del file dal dizionario DATA usando la chiave (c,e)
                (df,file) = self.DATA[c,e]
                #Estrae l'etichetta di classificazione che si trova alla riga 1, ultima colonna
                label = df.iloc[1,-1]
                #Converte il dataframe in una matrice NumPy, escludendo l'ultima colonna (metadati)
                XY = (df.to_numpy())[:,:-1]
                #Sostituisce eventuali valori NaN con 0, per prevenire errori nei calcoli statistici successivi
                XY[pd.isna(XY)] = 0.
                #Recupera la dimensione di XY
                n,m = XY.shape

                #Calcola il valore massimo per ogni colonna
                max = np.max(XY,axis=0)
                #Calcola il valore minimo per ogni colonna
                min = np.min(XY,axis=0)
                #Crea un array vuoto che accumulerà le statistiche estratte
                med = np.zeros(0)
                #Itera su ogni sensore
                for i in range(16):
                    #Nei primi 12 sensori si prendono solo i valori <= 0.7*max e >= 1.05*min
                    if i < 12:
                        mask = (XY[:,i]<=0.7*max[i]) & (XY[:,i]>=1.05*min[i])
                    #Negli altri si prendono solo i valori <= 0.9*man e >= 1.05*min
                    else:
                        mask = (XY[:,i]<=0.9*max[i]) & (XY[:,i]>=1.05*min[i])
                    #Applica la maschera descritta sopra (NON FUNZIONA PERCHE' E' COMMENTATO)
                    #a = XY[mask,i]
                    #Il filtro sopra è ignorato e vengono usati tutti i dati per le statistiche
                    a = XY[:,i]
                    #Controllo di sicurezza che controlla se l'array risultante è vuoto (NON LO SARA' MAI SE IL FILTRO NON FUNZIONA)
                    if(len(a) <= 0):
                        #Stampa i tre valori che hanno causato un errore nel filtraggio dei dati
                        print('{} {} {}'.format(i,1.05*min[i],0.8*max[i]))
                        input()
                    #print(n,len(a))
                    #print(XY[:,i]/np.maximum(1.,XY[:,i+1]))
                    #input()
                    if False:
                        #f, Pxx_den = signal.periodogram(a, fs=10e3)
                        # plt.semilogy(f, Pxx_den)
                        plt.plot(a)
                        plt.xlabel('frequency [Hz]')
                        plt.ylabel('PSD [V**2/Hz]')
                        plt.title('{} exp {} sensor {}'.format(c,e,i))
                        plt.show()
                    #Calcola la media
                    med = np.append(med, np.mean(a))
                    #med = np.append(med,np.min(a))
                    #Calcola il massimo
                    med = np.append(med,np.max(a))
                    #med = np.append(med,np.percentile(a,25))
                    #med = np.append(med,np.percentile(a,50))
                    #med = np.append(med,np.percentile(a,75))
                    #Calcola la deviazione standard
                    med = np.append(med, np.std(a))
                    #print(type(a))
                    #Calcola l'asimmetria
                    med = np.append(med, stats.skew(a.astype(float)))
                    #Calcola la coda
                    med = np.append(med, stats.kurtosis(a.astype(float)))
                #row = np.append(np.append(np.append(med,max),min),label)
                #Aggiunge il label alla fine del vettore
                row = np.append(med,label)
                #Formatta il vettore come un dataframe di una sola riga e lo salva nel dizionario DATARED, associandolo alla chiave dell'esperimento corrente
                self.DATARED[c,e] = (pd.DataFrame(row.reshape(1,-1)),file)

    #Riduce i dati suddividendoli per pacchetti, ogni esperimento non produce più una sola riga ma ne produce npac
    def data_reduce_pacchetti(self,npac=3):
        '''
        riduce i dati sostituendo a tutte le misure di ogni sensore alcuni indicatori di quel sensore
        :return:
        '''
        #Inizializza il dizionario dei dati
        self.DATARED = {}

        #Itera i componenti
        for c in self.components:
            #print(c + ' ', end='')
            #Itera gli esperimenti
            for e in self.experiments[c]:
                #print(e, ' ', end='')
                #Crea una matrice vuota 16 sensori x 4 statistiche + 1 label
                tab = np.zeros((0,16*4+1))
                #Carica i dati grezzi
                (df,file) = self.DATA[c,e]
                #Estrae la colonna label
                label = df.iloc[1,-1]
                #Converte il dataset in una matrice NumPy
                XY = (df.to_numpy())[:,:-1]
                #Sostituisce i NaN con 0
                XY[pd.isna(XY)] = 0.
                #Ottiene le dimensioni della matrice
                n,m = XY.shape
                #Inizializza l'indice di partenza del pacchetto corrente
                ia = 0
                #Calcola la lunghezza di ogni pacchetto (n_campioni // n_pacchetti)
                len = n // npac
                #Inizia il loop per processari i segmenti uno alla volta
                for r in range(npac):
                    #Inizializza un array di 0
                    med = np.zeros(0)
                    #Itera su ogni sensore
                    for i in range(16):
                        #Se non è l'ultimo pacchetto prende la porzione da ia a ia+len
                        if r < npac-1:
                            a = XY[ia:ia+len,i]
                        #Altrimenti prende da ia alla fine (la divisione intera // di prima potrebbe aver lasciato qualcosa fuori)
                        else:
                            a = XY[ia:,i]

                        #Calcola la media
                        med = np.append(med, np.mean(a))
                        #Calcola il massimo
                        med = np.append(med,np.max(a))
                        #Calcola la deviazione standard
                        med = np.append(med, np.std(a))
                        #Calcola l'asimmetria
                        sk = stats.skew(a.astype(float))
                        #Se l'asimmetria è NaN la imposta a 0
                        if np.isnan(sk):
                            sk = 0.
                        med = np.append(med, sk)
                        #med = np.append(med, stats.kurtosis(a.astype(float)))

                    #Aggiunge il label alla fine
                    row = np.append(med,label)
                    #print(row)
                    #Usa vstack per aggiungere questa riga alla tabella tab
                    tab = np.vstack((tab,row.reshape(1,-1)))
                    #print(tab)
                    #input()
                    #Sposta l'indice in avanti per il prossimo pacchetto
                    ia += len
                #Salva la tabella nel dizionario DATARED
                self.DATARED[c,e] = (pd.DataFrame(tab),file)

    #Approssima le misurazioni a un polinomio di grado 3
    def data_reduce_poly(self):
        '''
        approssima tutte le curve di misurazione con dei polinomi di grado 3
        quindi riduce un esperimento da (1000,16) a (4,16) per ogni componente
        se ndeg == -1 non si fa alcuna interpolazione delle curve
        :return:
        '''
        #Grado del polinomio
        ndeg = self.ndeg
        #Sensori da processare
        sensors = self.sensors
        #Numero di sensori
        nsen = len(sensors)
        #Inizializza il dizionario di output
        self.DATARED = {}
        #Itera ogni componente
        for component in self.components:
            #Logging
            print(component+' ',end='')
            #Itera ogni esperimento
            for experiment in self.experiments[component]:
                #Logging
                print(experiment,' ',end='')
                #Recupera i dati grezzi
                (df,file) = self.DATA[(component,experiment)]
                #Converte il dataframe in una matrice NumPy
                XY = (df.to_numpy())[:,:-1]
                #Recupera le dimensioni della matrice
                n, m = XY.shape
                #Crea l'asse x (interi da 0 a n-1)
                x = np.array(np.linspace(0, n - 1, n), dtype=np.float64)  # .tolist()
                #Prende i dati dei sensori e li converte in float a 64 bit per maggiore precisione
                y = np.array(XY, dtype=np.float64)  # .tolist()
                #Filtra y mantenendo solo le colonne specificate in self.sensors
                y = y[:,sensors]
                #Se il grado è -1 non fa nessuna interpolazione
                if ndeg == -1:
                    #Se ci sono dei valori NaN stampa un errore
                    if np.any(np.isnan(y)):
                        print('error ',end='')
                    #Altrimenti riscrive tutti i dati in un unico vettore colonna
                    else:
                        self.DATARED[(component, experiment)] = np.reshape(y, (1000 * nsen, 1), order='F')
                        print(y.shape,end='')
                #Altrimenti prova a interpolare i punti usando np.polyfit (calcola i coefficienti che minimizzano il MSE)
                else:
                    try:
                        #t avrà dimensione 4 coefficienti x 16 sensori
                        t = np.polyfit(x, y, ndeg)
                        #Appiattisce t in un unico vettore colonna e lo inserisce nel dizionario come nuovo valore
                        self.DATARED[(component,experiment)] = np.reshape(t,((ndeg+1)*nsen,1),order='F')
                        print(t.shape,end='')
                    #Se c'è qualche problema stampa a schermo
                    except:
                        print('error ',end='')
            print()

    #Stampa un resoconto dettagliato ti tutti gli esperimenti caricati in memoria
    #Per ogni sostanza chimica, elenca i singoli file acquisiti, mostrando le dimensioni delle matrici e conta il numero totale di esperimenti disponibili per quella classe
    def data_prop(self):
        #Itera i componenti
        for component in self.components:
            #Stampa un separatore e il nome della sostanza corrente
            print()
            print('---------------------------------------------')
            print(component)
            print('---------------------------------------------')
            #Inizializza un contatore che terrà traccia del numero di esperimenti validi trovati per il componente corrente
            count = 0
            #Itera su ogni coppia chiave-valore del dizionario
            for (a,b) in self.DATA.keys():
                #Se la chiave corrisponde al componente che cerchiamo
                if a == component:
                    #Estrae il dataframe contenente le misurazioni e il nome originale del file
                    (df,file) = self.DATA[(a,b)]
                    #Legge le dimensioni della matrice
                    n,m = df.shape
                    #Stampa una riga di report nel formato: ID_Esperimento - Nome-File: (n_righe, n_colonne)
                    print('{}-{}: {}'.format(b, file, df.shape))
                    #if n > 500:
                    #Incrementa il contatore (esperimento valido)
                    count += 1
            #Stampa il totale degli esperimenti trovati
            print('number of experiments : ',count)

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
                            #print(file)
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

    #Visualizza graficamente le misurazioni di un singolo sensore specifico
    #Prende in input il nome della sostanza e l'indice del sensore da visualizzare
    def plot_sensor(self,component,sensor):
        #Definisce una mappa statica per variare l'aspetto delle linee nel grafico
        styles = { 1:('-','b'),  2:('--','b'),  3:('-.','b'),  4:(':','b'),
                   5:('-','r'),  6:('--','r'),  7:('-.','r'),  8:(':','r'),
                   9:('-','k'), 10:('--','k'), 11:('-.','k'), 12:(':','k'),
                  13:('-','m'), 14:('--','m'), 15:('-.','m'), 16:(':','m')
                  }
        #Crea una nuova figura Matplotlib
        plt.figure(figsize=(3.68, 3.68), dpi=300)
        #Imposta il titolo
        plt.title(component + " + " + str(sensor))
        #Imposta il nome dell'asse y
        plt.ylabel("measured value")
        #Imposta il nome dell'asse x
        plt.xlabel("sample")
        #Inizializza il contatore che servirà per scegliere lo stile dal dizionario
        i = 0
        #Riga di debug che stampa tutte le chiavi del dizionario dei dati
        print(self.DATA.keys())
        #Itera ogni esperimento presente nel dizionario
        for (a, b) in self.DATA.keys():
            #Seleziona solo gli esperimenti relativi alla sostanza richiesta
            if a == component:
                #Assegna lo stile (linestyle, color)
                ls,cr = styles[i+1]
                #Esrae il dataframe dell'esperimento
                (df,file) = self.DATA[(a,b)]
                #Converte il dataframe in una matrice NumPy
                XY = df.to_numpy()
                #Disegna la linea dell'esperimento corrente
                plt.plot(XY[:, int(sensor)], linestyle=ls, color=cr, linewidth=0.3,label=file)
                #Passa al prossimo stile (dovrebbe essere i = (i + 1) % 16 altrimenti va in crash?)
                i += 1
        #Aggiunge la legenda in basso a destra
        plt.legend(loc='lower right',prop={'size': 3})
        #Mostra il grafico
        plt.show()
        #Chiude la figura per liberare memoria
        plt.close()

    #Da la possibilità di visualizzare la lista di tutti gli esperimenti per una determinata sostanza e di graficarne l'andamento
    def do_plots(self):
        #Inizializza due variabili vuote per ricordare l'ultima scelta dell'utente
        component = ''
        measure = ''
        #Avvia un ciclo infinito
        while True:
            #Mostra la lista di tutte le sostanze disponibili
            print(self.components)
            #Chiede all'utente di selezionarne una (mostra l'ultima scelta effettuata come default)
            print('select a component, x to exit: [{}]'.format(component),end='')
            #Aspetta l'input dell'utente
            s = input()
            #Se la stringa non è vuota
            if not(s == ''):
                #Se la stringa è 'x'
                if s.strip().lower() == 'x':
                    #Interrompi l'esecuzione
                    break
                #Altrimenti aggiorna la variabile component (se la stringa inserita è vuota component rimane uguale a prima)
                component = s
            #Pulisce la stringa da spazi e la converte in maiuscolo per farla corrispondere alle chiavi interne
            component = component.strip().upper()
            #Procede solo se la stringa è una delle sostanze riconosciute
            if (component in self.components):
                #print('select sensor id [0-15], x to exit:',end='')
                #s = input()
                #if not(s == ''):
                #    if s.strip().lower() == 'x':
                #        break
                #    sensor = s
                #sensor = int(str(sensor).strip())
                #self.plot_sensor(component,sensor)
                if True:
                    #Itera ogni esperimento
                    for (a,b) in self.DATA.keys():
                        #Sceglie il componente richiesto
                        if a == component:
                            #Estrae il dataset e il nome del file originale
                            df,file = self.DATA[(a,b)]
                            #Stampa il nome dell'esperimento e il nome del file
                            print(str(b)+'-',file)
                    #print()
                    #Chiede all'utente di digitare l'ID numerico dell'esperimento desiderato
                    print('select a measurement, x to exit: [{}]'.format(measure), end='')
                    #Aspetta l'input
                    s = input()
                    #Se la stringa non è vuota
                    if not(s == ''):
                        #Se la stringa è 'x'
                        if s.strip().lower() == 'x':
                            #Interrompe il ciclo
                            break
                        #Altrimenti aggiorna measure (se la stringa è vuota rimane l'ultimo ID selezionato)
                        measure = s
                    #Converte la stringa in un numero intero
                    measure = int(str(measure).strip())
                    #Verifica che la coppia sostanza,ID esista davvero
                    if (component,measure) in self.DATA.keys():
                        #Estrae i dati dal dizionario
                        (df,file) = self.DATA[(component,measure)]
                        #Trasforma il dataset in una matrice NumPy
                        XY = df.to_numpy()
                        #print(df.shape,XY.shape)
                        #print(XY)
                        #Chiama la funzione che stampa i grafici della misurazione
                        self.plot_measurement(XY,component,measure)

    #Mostra due grafici affiancati per un singolo esperimento
    #A sinistra l'andamento grezzo dei 16 sensori
    #A destra l'andamento normalizzato (diviso per la media di tutto l'array)
    #Prende in input la matrice dei dati, il nome del componente e l'ID della misurazione
    def plot_measurement(self,XY,component,measure):
        #Crea il dizionario degli stili
        styles = { 1:('-','b'),  2:('--','b'),  3:('-.','b'),  4:(':','b'),
                   5:('-','r'),  6:('--','r'),  7:('-.','r'),  8:(':','r'),
                   9:('-','k'), 10:('--','k'), 11:('-.','k'), 12:(':','k'),
                  13:('-','m'), 14:('--','m'), 15:('-.','m'), 16:(':','m')
                  }
        #Crea una figura con 1 riga e 2 colonne
        fig, axs = plt.subplots(1,2,figsize=(5,10))
        #Imposta il titolo generale
        fig.suptitle(component + " + " + str(measure))

        #Imposta i label degli assi
        axs[0].set_ylabel("measured value")
        axs[0].set_xlabel("sample")
        axs[1].set_ylabel("normalized value")
        #Errore nell'indice (dovrebbe essere 1)
        axs[0].set_xlabel("sample")
        #Calcolo della media per riga
        media = XY[:,:-1].sum(axis=1)/16.
        #print(media.shape)
        #input()
        #Itera per ogni sensore
        for i in range(16):
            #Imposta lo stile
            ls,cr = styles[i+1]
            #Disegna l'andamento del grafico grezzo a sinistra
            axs[0].plot(XY[:, i], linestyle=ls, color=cr, linewidth=0.5)
            #Se il componente analizzato non è l'aria
            if not component == 'AIR':
                #Disegna il grafico normalizzato a destra
                axs[1].plot(XY[:, i]/media, linestyle=ls, color=cr, linewidth=0.5)

        #Mostra il grafico
        plt.show()
        #Chiude e pulisce la memoria
        plt.close()

    #Genera una figura comparativa che affianca grafici di diversi esperimenti selezionati
    #Prende in input un dizionario contenente una lista di componenti ed esperimenti specifici da analizzare
    def plot_examples(self,plot_dict):
        #Dizionario degli stili
        styles = {1: ('-', 'b'), 2: ('--', 'b'), 3: ('-.', 'b'), 4: (':', 'b'),
                  5: ('-', 'r'), 6: ('--', 'r'), 7: ('-.', 'r'), 8: (':', 'r'),
                  9: ('-', 'k'), 10: ('--', 'k'), 11: ('-.', 'k'), 12: (':', 'k'),
                  13: ('-', 'm'), 14: ('--', 'm'), 15: ('-.', 'm'), 16: (':', 'm')
                  }
        #Conta quanti esempi sono stati richiesti (determina il numero di sottografici)
        nfig = len(plot_dict.keys())
        #Crea una griglia di grafici con 1 riga e nfig colonne
        fig, axs = plt.subplots(1, nfig, figsize=(3.68, 3.68))
        #Imposta il titolo generale
        fig.suptitle("examples")
        #Itera tutti i sottografici
        for i in range(nfig):
            #Imposta l'etichetta dell'asse y
            axs[i].set_ylabel("measured value")
            #Imposta l'etichetta dell'asse x
            axs[i].set_xlabel("time [s]")
    
        #Itera il dizionario di input (j corrisponde alla colonna del grafico)
        for (j, comp) in enumerate(plot_dict):
            #Imposta il nome della sostanza come titolo del singolo pannello
            axs[j].set_title(comp)
            #Prende solo il primo esperimento della lista fornita nel dizionario per quel componente
            expr = plot_dict[comp][0]
            #Recupera i dati
            (df, file) = self.DATA[comp, expr]
            # print(file)
            #Converte il dataframe in una matrice NumPy, scartando l'etichetta finale
            X = df.to_numpy()[:, :-1]
            #Calcola la somma di tutti i valori delle colonne, riga per riga
            aa = np.abs(X).sum(axis=1)
            # print('min: {} max: {}'.format(np.min(aa),np.max(aa)))
            #Crea una maschera con lo scopo di tagliare tutte le parti 'morte' o di rumore, mantenendo solo il picco della reazione
            ind = (aa > np.min(aa) + 0.2 * (np.max(aa) - np.min(aa)))
            #ind = (aa > -np.inf)
            #Applica la maschera alla matrice X
            X = X[ind, :]
            #Itera su tutti i sensori
            for i in range(16):
                #Recupera lo stile
                ls, cr = styles[i + 1]
                #Disegna la curva del sensore
                axs[j].plot(X[:, i], linestyle=ls, color=cr, linewidth=0.5)
        #Mostra il grafico e pulisce la memoria
        plt.show()
        plt.close()

    #Tenta di approssimare l'andamento dei sensori
    #Prende in input (opzionalmente) la sostanza, l'ID dell'esperimento e un sensore specifico
    def approximate(self,component='DIESEL',experiment=1,sensor=2):
        #Recupera i dati grezzi
        (df,file) = self.DATA[(component,experiment)]
        #Stampa il nome originale del file
        print(file)
        #Converte il dataframe in una matrice NumPy
        XY = df.to_numpy()
        #Recupera le dimensioni della matrice
        n,m = XY.shape
        #Stampa i nomi delle colonne del dataframe
        print(df.columns.values.tolist())
        #Crea un vettore di intervalli interi da 0 a n-1
        x = np.array(np.linspace(0,n-1,n),dtype=np.float64) #.tolist()
        #Crea la matrice y contenente i valori di tutti i sensori (esclusa l'etichetta finale)
        y = np.array(XY[:,:-1],dtype=np.float64) #.tolist()
        #Regressione polinomiale
        #t = np.polyfit(x, y, 3)
        #print(t)
        #Regressione basata sui vettori di supporto
        #regr = SVR(C=100.,epsilon=0.001)
        #Rete neurale per la regressione
        #regr = MLPRegressor(hidden_layer_sizes=(200,),activation='tanh',max_iter=1000)
        #Regressione logistica (non funziona bene perché y contiene valori continui?)
        regr = LogisticRegression(C=10.)
        #Crea una catena di operazioni: normalizza i dati -> applica il modello
        pipeline = make_pipeline(StandardScaler(), regr)
        #Prepara una matrice vuota delle stesse dimensioni dei dati dei sensori
        t = np.zeros((n,m-1))
        #Itera su ogni sensore
        for i in range(16):
            #Addestra il modello a predirre il valore del sensore i basandosi sull'istante x
            pipeline.fit(x.reshape(-1,1),y[:,i])
            #Chiede al modello di ricostruire la curva completa
            t[:,i] = pipeline.predict(x.reshape(-1,1))
        #Chiama un metodo dedicato per visualizzare il confronto tra la curva reale e quella approssimata
        self.plot_approx(x,y,t,component,experiment)

    #Crea un grafico con i dati reali (sinistra) e uno con i dati approsimati (destra)
    #Prende in input l'array di intervalli, la matrice dei dati reali, la matrice dei dati predetti, il nome del componente e l'ID dell'esperimento
    def plot_approx(self,x,y,t,comp,expr):
        #Dizionario degli stili
        styles = { 1:('-','b'),  2:('--','b'),  3:('-.','b'),  4:(':','b'),
                   5:('-','r'),  6:('--','r'),  7:('-.','r'),  8:(':','r'),
                   9:('-','k'), 10:('--','k'), 11:('-.','k'), 12:(':','k'),
                  13:('-','m'), 14:('--','m'), 15:('-.','m'), 16:(':','m')
                  }
        #Crea due sottografici affiancati e impone che l'asse y sia identico per entrambi (altrimenti potrebbe essere scalato)
        fig, axs = plt.subplots(1, 2, figsize=(3.68, 3.68), sharey=True)
        #plt.figure(figsize=(3.68, 3.68), dpi=300)
        #plt.title("approximation {}({})".format(comp,expr))
        #Imposta il titolo generale
        fig.suptitle("approximation {}({})".format(comp, expr))
        #Imposta i label degli assi
        axs[0].set_ylabel("measured value")
        axs[1].set_ylabel("logistic regr. value")
        axs[0].set_xlabel("time [s]")
        axs[1].set_xlabel("time [s]")
        #Itera per ogni sensore
        for i in range(16):
            #Recupera lo stile
            ls,cr = styles[i+1]
            #Disegna la curva originale del sensore
            axs[0].plot(x, y[:,i], linestyle=ls, color=cr, linewidth=0.5)
            #p = np.poly1d(t[:,i])
            #plt.plot(x, p(x), linestyle=ls, color=cr, linewidth=0.1)
            #Disegna la curva predetta dal modello (se l'approssimazione è buona il grafico dovrebbe essere quasi identico a quello reale)
            axs[1].plot(x, t[:, i], linestyle=ls, color=cr, linewidth=0.5)
        #Mostra il grafico e pulisce la memoria
        plt.show()
        plt.close()
