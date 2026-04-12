import os

from class_factory_LIGHT import SCA_NRM_data
from class_factory_LIGHT import experiment
from contextlib import redirect_stdout
import requests
        

if __name__ == "__main__":

    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        reduce = False
        SCA = SCA_NRM_data(help=False,verbose=False)

        nmin = 4
        c = SCA.components[0]
        e = SCA.experiments[c][0]
        (a, b) = SCA.DATA[(c, e)]
        nmis, nsensors = a.to_numpy()[:, :-1].shape

        sensors = range(nsensors)

        EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components='ALL',feature=True)

    print("[Debug] Dati caricati")

    #normalized=False
    #EXP.prova_fast_classifiers(normalized=False,balance=False)
    #EXP.prova_fast_classifiers(normalized=False,balance=True)
    #EXP.prova_mlp(normalized=False,balance=False)
    #EXP.prova_mlp(normalized=False,balance=True)
    #EXP.prova_slow_classifiers(normalized=False,balance=False)
    #EXP.prova_slow_classifiers(normalized=False,balance=True)
    
    #normalized=True
    #EXP.prova_fast_classifiers(normalized=True,balance=False)
    #EXP.prova_fast_classifiers(normalized=True,balance=True)
    #EXP.prova_mlp(normalized=True,balance=False)
    #EXP.prova_mlp(normalized=True,balance=True)
    #EXP.prova_slow_classifiers(normalized=True,balance=False)
    #EXP.prova_slow_classifiers(normalized=True,balance=True)
    
    #EXP.prova_votazione(normalized=True,balance=True)
    
    #EXP.prova_top4_cross_validation(n_iterazioni=10, nome='', undersampling=True)
        
    #gruppi = {
    #    'COV':          ['ACETONE','ETHANOL','BIOETHANOL','ISOPROPANOL','NITROMETHANE'],
    #    'Idrocarburi':  ['METHANE','BUTANE','GASOLINE','LIGHTER_FLUID','DIESEL','KEROSENE'],
    #    'Acidi':        ['ACETIC_ACID','FORMIC_ACID','PHOSPHORIC_ACID'],
    #    'Inorganici':   ['AMMONIA','AMMONIUM_CHLORIDE','CALCIUM_NITRATE','SODIUM_HYDROXIDE','UREA'],
    #    'Acquosi':      ['RED_WINE','APPLE_VINEGAR','BALSAMIC_VINEGAR'],
    #    'Baseline':     ['AIR','WATER_VAPOR','HYDROGEN_PEROXIDE']
    #}
    #
    #for nome_gruppo in gruppi.keys():
    #    EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=gruppi[nome_gruppo])
    #    EXP.prova_top4_cross_validation(n_iterazioni=10, nome=nome_gruppo, undersampling=True)
    
    #selezione_1 = ['ACETONE','ETHANOL','BIOETHANOL','NITROMETHANE',         #'ISOPROPANOL'
    #              'METHANE','BUTANE','LIGHTER_FLUID','DIESEL',              #'GASOLINE','KEROSENE'
    #              'ACETIC_ACID','FORMIC_ACID','PHOSPHORIC_ACID', 
    #              'AMMONIA','CALCIUM_NITRATE','SODIUM_HYDROXIDE','UREA',    #'AMMONIUM_CHLORIDE'
    #              'RED_WINE','APPLE_VINEGAR','BALSAMIC_VINEGAR',
    #              'AIR','WATER_VAPOR'                                       #'HYDROGEN_PEROXIDE'
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_1)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_1',undersampling=True)

    #Risultati:
    #KNN:                           77.0% ± 9.7%
    #Random Forest:                 70.5% ± 12.2%
    #LightGBM:                      65.4% ± 11.3%
    #MLP Diamante (64, 128, 64):    75.4% ± 7.9%

    #selezione_2 = ['ACETONE', 'ETHANOL', 'BIOETHANOL', 'NITROMETHANE',      #'ISOPROPANOL'
    #               'METHANE', 'LIGHTER_FLUID', 'DIESEL',                    #'GASOLINE','KEROSENE', 'BUTANE'
    #               'ACETIC_ACID', 'FORMIC_ACID',                            #'PHOSPHORIC_ACID'
    #               'AMMONIA',                                               #'AMMONIUM_CHLORIDE','CALCIUM_NITRATE','SODIUM_HYDROXIDE','UREA'
    #               'RED_WINE', 'APPLE_VINEGAR', 'BALSAMIC_VINEGAR',
    #               'AIR', 'WATER_VAPOR'                                     #'HYDROGEN_PEROXIDE'
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_2)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_2',undersampling=True)

    #Risultati:
    #KNN:                           80.3% ± 12.0%
    #Random Forest:                 76.1% ± 13.8%
    #LightGBM:                      77.0% ± 11.8%
    #MLP Diamante (64, 128, 64):    92.9% ± 5.5%

    #selezione_3 = ['ACETONE', 'ETHANOL', 'BIOETHANOL', 'ISOPROPANOL', 'NITROMETHANE', 
    #               'METHANE', 'BUTANE', 'LIGHTER_FLUID', 'DIESEL',                         #'GASOLINE','KEROSENE',
    #               'ACETIC_ACID', 'FORMIC_ACID',                                           #'PHOSPHORIC_ACID'
    #               'AMMONIA',                                                              #'AMMONIUM_CHLORIDE','CALCIUM_NITRATE','SODIUM_HYDROXIDE','UREA'
    #               'RED_WINE', 'APPLE_VINEGAR', 'BALSAMIC_VINEGAR',
    #               'AIR', 'WATER_VAPOR', 'HYDROGEN_PEROXIDE'
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_3)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_3',undersampling=True)
    
    #Risultati:
    #KNN:                           70.5% ± 11.3%
    #Random Forest:                 64.0% ± 11.1%
    #LightGBM:                      62.3% ± 5.6%
    #MLP Diamante (64, 128, 64):    81.6% ± 11.1%
    
    #selezione_4 = ['ACETONE', 'ETHANOL', 'BIOETHANOL', 'ISOPROPANOL', 'NITROMETHANE', 
    #               'METHANE', 'LIGHTER_FLUID', 'DIESEL', 'KEROSENE' ,                       #'GASOLINE','BUTANE'
    #               'ACETIC_ACID', 'FORMIC_ACID',                                            #'PHOSPHORIC_ACID'
    #               'AMMONIA', 'SODIUM_HYDROXIDE', 'AMMONIUM_CHLORIDE', 'CALCIUM_NITRATE',   #'CALCIUM_NITRATE','UREA'
    #               'RED_WINE', 'APPLE_VINEGAR', 'BALSAMIC_VINEGAR',
    #               'AIR', 'WATER_VAPOR'                                                     #'HYDROGEN_PEROXIDE'
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_4)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_4',undersampling=True)
    
    #Risultati:
    #KNN:                           67.3% ± 11.0%
    #Random Forest:                 69.1% ± 14.3%
    #LightGBM:                      67.5% ± 8.2%
    #MLP Diamante (64, 128, 64):    82.3% ± 9.1%
    
    #selezione_5 = ['ACETONE', 'ETHANOL', 'BIOETHANOL', 'ISOPROPANOL', 'NITROMETHANE', 
    #               'METHANE', 'LIGHTER_FLUID', 'DIESEL', 'KEROSENE' ,                                       #'GASOLINE','BUTANE'
    #               'ACETIC_ACID', 'FORMIC_ACID', 'PHOSPHORIC_ACID',
    #               'AMMONIA', 'SODIUM_HYDROXIDE', 'AMMONIUM_CHLORIDE', 'CALCIUM_NITRATE',                   #'UREA'
    #               'RED_WINE', 'APPLE_VINEGAR', 'BALSAMIC_VINEGAR',
    #               'AIR'                                                                                    #'WATER_VAPOR','HYDROGEN_PEROXIDE'
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_5)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_5',undersampling=True)
    
    #Risultati:
    #KNN:                           65.0% ± 10.9%
    #Random Forest:                 64.1% ± 13.1%
    #LightGBM:                      62.6% ± 7.8%
    #MLP Diamante (64, 128, 64):    78.7% ± 7.9%
    
    #selezione_6 = ['ACETONE', 'ETHANOL', 'BIOETHANOL', 'NITROMETHANE',                                      #'ISOPROPANOL'
    #               'METHANE', 'LIGHTER_FLUID', 'DIESEL', 'KEROSENE', 'BUTANE',                              #'GASOLINE'
    #               'ACETIC_ACID', 'FORMIC_ACID', 'PHOSPHORIC_ACID',
    #               'AMMONIA', 'SODIUM_HYDROXIDE',                                                           #'UREA','CALCIUM_NITRATE', 'AMMONIUM_CHLORIDE'
    #               'RED_WINE', 'APPLE_VINEGAR', 'BALSAMIC_VINEGAR'
    #                                                                                                        #'WATER_VAPOR','HYDROGEN_PEROXIDE','AIR' 
    #            ]
    #EXP = experiment(SCA,sensors=sensors,nmin=nmin,soglia=0.25,TR='ANY',TS='ANY',debug=False,components=selezione_6)
    #EXP.prova_top4_cross_validation(n_iterazioni=10,nome='selezione_6',undersampling=True)
    
    #Risultati:
    #KNN:                           81.5% ± 14.7%
    #Random Forest:                 75.4% ± 12.5%
    #LightGBM:                      81.0% ± 11.3%
    #MLP Diamante (64, 128, 64):    83.2% ± 9.8%
    
    #EXP.prova_configurazioni_mlp()
    #EXP.prova_top4_cross_validation(n_iterazioni=10, nome='256', undersampling=True)
    #EXP.prova_strati_idrocarburi()
    
    #try:
    #    EXP.prova_configurazioni_mlp()
    #    message = "Completati2"
    #except Exception as e:
    #    message = f"Errore2: {e}"
    #requests.post("http://ntfy.sh/prove_mlp", data=message.encode("utf-8"))
    
    