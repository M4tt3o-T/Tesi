import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Importiamo le configurazioni
import config

class DataManager:
    
    def __init__(self):
        self.components = config.COMPONENTS
        self.filenamepickle = config.CACHE_FILE
        self.traindir = config.TRAIN_DIR
        self.DATA = {}
        self.experiments = {}
        self.processed_data = {}

        # Caricamento iniziale
        if os.path.isfile(self.filenamepickle):
            self.extract_data(export=False)
        else:
            self.extract_data(export=True)

        # Raggruppamento esperimenti
        for component in self.components:
            expr = [b for (a, b) in self.DATA.keys() if a == component]
            self.experiments[component] = expr

    def extract_data(self, export=True):
        self.DATA = {}
        if export:
            for component in self.components:
                print(component + ' ', end='')
                traindata = os.path.join(self.traindir, component)

                if os.path.isdir(traindata):
                    expid = 1
                    for file in os.listdir(traindata):
                        if file.endswith(".csv"):
                            print('exp' + str(expid) + ' ', end='')
                            self.DATA[(component, expid)] = (pd.read_csv(os.path.join(traindata, file), sep=';'), file)
                            expid += 1
                print()

            with open(self.filenamepickle, 'wb') as filepickle:
                pickle.dump(self.DATA, filepickle)
        else:
            with open(self.filenamepickle, 'rb') as file:
                self.DATA = pickle.load(file)

    def preprocess_data(self, feature=True, soglia=config.SOGLIA, sensors=config.SENSORS):
        """Applica la soglia per il rumore e opzionalmente calcola le feature orizzontali."""
        self.processed_data = {}
        
        for (c, e), (df, filename) in self.DATA.items():
            t_raw = df.to_numpy()[:, sensors].astype(np.float64)
            t_raw[pd.isna(t_raw)] = 0.

            if feature:
                row_mean = np.mean(t_raw, axis=1).reshape(-1, 1)
                row_std = np.std(t_raw, axis=1).reshape(-1, 1)
                row_max = np.max(t_raw, axis=1).reshape(-1, 1)
                t_matrix = np.hstack((t_raw, row_mean, row_std, row_max)).astype(np.float64)
            else:
                t_matrix = t_raw

            # Applicazione della soglia basata sui dati grezzi
            aa_abs = np.abs(t_raw).sum(axis=1)
            if np.max(aa_abs) > np.min(aa_abs):
                ind = (aa_abs > np.min(aa_abs) + soglia * (np.max(aa_abs) - np.min(aa_abs)))
                t_matrix = t_matrix[ind, :]

            self.processed_data[(c, e)] = t_matrix
            
        # Aggiorna il numero finale di feature per usi futuri
        return t_matrix.shape[1] if bool(self.processed_data) else len(sensors)

    def split_experiments(self, c, nmin=config.NMIN):
        """Divide gli esperimenti in NUOVI, VECCHI e PICCOLI."""
        experiments = {'NUOVI': [], 'VECCHI': [], 'PICCOLI': []}
        if c not in self.components:
            raise ValueError(f"Component '{c}' not found.")

        for e in self.experiments[c]:
            (_, filename) = self.DATA[(c, e)]
            t_filtered = self.processed_data.get((c, e), np.array([]))
            
            if t_filtered.shape[0] <= nmin:
                experiments['PICCOLI'].append(e)
            else:
                if 'slow' in filename.lower() or 'checkair' in filename.lower():
                    experiments['NUOVI'].append(e)
                else:
                    experiments['VECCHI'].append(e)
        return experiments

    def define_train_test_exp(self, c, exclude, tr='SLOW', ts='ALL'):
        """Assegna gli ID degli esperimenti al train o al Test set in base alla strategia."""
        experiments = self.split_experiments(c)
        list_slow_exp = experiments['NUOVI']
        list_stat_exp = experiments['VECCHI'] if c in config.LISTA_AMMESSI_STAT else []

        nslow, nstat = len(list_slow_exp), len(list_stat_exp)
        train, test = [], []

        if nstat + nslow == 0:
            return train, test

        #Divide train e test set a seconda delle richieste
        if tr == 'SLOW' and ts == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif tr == 'SLOW' and ts == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif tr == 'SLOW' and ts == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,[]]
            ntr = [nslow,0]
        elif tr == 'SLOW' and ts == 'ANY':
            lts = [list_slow_exp, list_stat_exp]
            nts = [nslow, nstat]
            ltr = [list_slow_exp, []]
            ntr = [nslow, 0]
        elif tr == 'STAT' and ts == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif tr == 'STAT' and ts == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif tr == 'STAT' and ts == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [[],list_stat_exp]
            ntr = [0,nstat]
        elif tr == 'STAT' and ts == 'ANY':
            lts = [list_slow_exp, list_stat_exp]
            nts = [nslow, nstat]
            ltr = [[], list_stat_exp]
            ntr = [0, nstat]
        elif tr == 'ALL' and ts == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ALL' and ts == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ALL' and ts == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ALL' and ts == 'ANY':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ANY' and ts == 'SLOW':
            lts = [list_slow_exp,[]]
            nts = [nslow,0]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ANY' and ts == 'STAT':
            lts = [[],list_stat_exp]
            nts = [0,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ANY' and ts == 'ALL':
            lts = [list_slow_exp,list_stat_exp]
            nts = [nslow,nstat]
            ltr = [list_slow_exp,list_stat_exp]
            ntr = [nslow,nstat]
        elif tr == 'ANY' and ts == 'ANY':
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

        return train, test

    def generate_trainset(self, dict_comp, exclude, normalized=True, balance=True, undersampling=True, tr='ALL', ts='ALL', random_state=100):
        """Assembla le matrici X e y pronte per il machine learning."""
        np.random.seed(random_state)
        X_list, Xts_list, y_list, yts_list, exp_ids_ts_list = [], [], [], [], []
        cardinality = []

        # Determina il numero di sensori/feature guardando il primo elemento processato
        nsensors = next(iter(self.processed_data.values())).shape[1]

        for comp, comp_list in dict_comp.items():
            if comp == 'OTHERS':
                continue
                
            num, numts = 0, 0
            for c in comp_list:
                ltr, lts = self.define_train_test_exp(c, exclude[c], tr=tr, ts=ts)
                
                # train set
                for e in ltr:
                    t = self.processed_data[(c, e)]
                    if normalized:
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))
                    
                    X_list.append(t)
                    y_list.extend([comp] * t.shape[0])
                    num += t.shape[0]

                # Test set
                for e in lts:
                    t = self.processed_data[(c, e)]
                    if normalized:
                        aa_mean = t.sum(axis=1) / nsensors
                        aa_mean[np.abs(aa_mean) < 0.1] = 1.
                        t = t / np.tile(np.reshape(aa_mean, (-1, 1)), (1, nsensors))
                    
                    Xts_list.append(t)
                    yts_list.extend([comp] * t.shape[0])
                    exp_ids_ts_list.extend([f"{comp}_{e}"] * t.shape[0])
                    numts += t.shape[0]

            cardinality.append({'comp': comp, 'card': num})

        X = np.vstack(X_list) if X_list else np.array([])
        Xts = np.vstack(Xts_list) if Xts_list else np.array([])
        y = np.array(y_list)
        yts = np.array(yts_list)
        exp_ids_ts = np.array(exp_ids_ts_list)

        # Bilanciamento (Undersampling / Oversampling)
        if (undersampling or balance) and len(X) > 0:
            df_card = pd.DataFrame(cardinality)
            target_card = int(df_card['card'].mean())
            
            X_balanced, y_balanced = [], []
            for _, row in df_card.iterrows():
                comp = row['comp']
                current_card = row['card']
                class_indices = np.where(y == comp)[0]
                
                if len(class_indices) > 0:
                    if current_card > target_card and undersampling:
                        sampled_indices = np.random.choice(class_indices, size=target_card, replace=False)
                    elif current_card < target_card and balance:
                        sampled_indices = np.random.choice(class_indices, size=target_card, replace=True)
                    else:
                        sampled_indices = class_indices
                        
                    X_balanced.append(X[sampled_indices])
                    y_balanced.append(y[sampled_indices])
                    
            X = np.vstack(X_balanced)
            y = np.concatenate(y_balanced)
            X, y = shuffle(X, y, random_state=random_state)

        return X, Xts, y, yts, exp_ids_ts
    
if __name__ == "__main__":
    data = DataManager()