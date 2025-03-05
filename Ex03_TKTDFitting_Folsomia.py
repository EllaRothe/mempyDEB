import pandas as pd
import numpy as np
import os
from copy import deepcopy
import warnings
import pprint 

from ModelFitting import *

from mempyDEB.DEBODE.simulators import *
from mempyDEB.DEBODE.defaultparams import *

from Ex03_DEBFitting_Folsomia import collembola_length_to_weight

# Konstanten

EMB_DEV_TIME = 2 # geschätzte embryonalentwicklungszeit (realistisch für Daphnia)
S_MAX_REFERENCE = 388.4 # maximale Strukturelle Masse, die durch die Anfangswerte impliziert ist
EXPOSURES = [0,100,500,1500,2000]

def calc_S_max(spc: dict):
    """
    Berechnung maximaler struktureller Masse auf Basis der DEB-Parameter.
    """ 

    return np.power((spc['kappa'] * spc['Idot_max_rel'] * spc['eta_IA_0']) / spc['k_M_0'], 3)

def load_data():
    """
    Einlesen der Kontrolldaten für Folsomia
    """
    
    data = pd.read_csv('folsomia_temperature_cadmium_growth_tidy25.csv', header = 5)
    #data = data[data['T_cels']==15]
    data = data.assign(S = collembola_length_to_weight(data.length_mm))
    data.rename(columns = {'C_F':'C_W'}, inplace = True)

    return data

def plot_data(data):
    """
    Plotten der Kontrolldaten für Folsomia. 
    Gibt ein `fig, ax`-Tuple zurück.
    """

    fig, ax = plt.subplots(ncols = 5, nrows = 2, figsize = (20,8), sharey = True)

    for (i,C_W) in enumerate(data.C_W.unique()):

        ax[0,i].set(title = f'{C_W} mg/kg')
        obs = data.loc[lambda df : df.C_W==C_W]

        sns.lineplot(obs.loc[obs['T_cels']==15], x = 't_day', y = 'S', ax = ax[0,i], marker = 'o', color = 'forestgreen')
        sns.lineplot(obs.loc[obs['T_cels']==20], x = 't_day', y = 'S', ax = ax[1,i], marker = 'o', color = 'mediumblue')
        
    ax[0,0].legend()
    [a.legend().remove() for a in np.ravel(ax)[1:]]
    ax[0,0].set_ylim(0, 25)
    ax[0,0].set(ylabel = "Struktur bei 15°C (mug)")
    ax[1,0].set(ylabel = "Struktur bei 20°C (mug)")

    sns.despine()
    plt.tight_layout()

    return fig,ax

def plot_sim(ax, sim):

    for (temp,T_cels) in enumerate(sim.T_cels.unique()):

        for (i,C_W) in enumerate(sim.C_W.unique()):
            df = sim.loc[(sim.C_W == C_W) & (sim.T_cels == T_cels)]
            sns.lineplot(df, x = 't_day', y = 'S', ax = ax[temp,i])

    return ax

#### Simulator-Funktion


def define_simulator(f: ModelFit):

    """
    Definition der Simulator-Funktion für DEB-Kalibrierung.
    """

    def simulator(theta: dict) -> tuple: # theta = rand(priors)

        p = deepcopy(f.defaultparams)
        p.spc.update(theta) # macht das gleiche wie p.spc['Idot_max_rel'] = theta['Idot_max_rel']
        S_max_theta = calc_S_max(p.spc) # implizierte maximale Struktur auf basis von theta
        zoom_factor_theta = S_max_theta / S_MAX_REFERENCE # zoom factor auf basis von theta
        
        # diese Parameter skalieren mit der neuen maximalen Größe

        p.spc['Idot_max_rel_emb'] *= zoom_factor_theta**(1/3)
        p.spc['X_emb_int'] *= zoom_factor_theta
        p.spc['S_p'] *= zoom_factor_theta
        list = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for temp in [288.15,293.15]:
                p.glb['T'] = temp
                prediction = constant_exposures(
                    simulate_DEBBase, p, EXPOSURES).rename({'t' : 't_day'}, axis = 1)
                prediction['T_cels'] = temp - 273.15
                list.append(prediction)

            prediction = pd.concat(list)

                # Berechnung der relative response
            prediction = pd.merge(
                prediction, 
                prediction.loc[prediction['C_W']==0], on = ['t_day'], 
                suffixes = ['', '_ref']).groupby(['t_day', 'C_W']).apply(
                    lambda gb : gb.assign(y_S = lambda gb : gb.S / gb.S_ref,))

            return prediction.drop(['C_W', 't_day'], axis = 1).reset_index()
            
    return simulator


def define_defaultparams():
    """
    Definition der Standard-Parameterwerte für DEB-Modellkalibrierung. 

    Modellwährung ist mugC. 
    """

    glb = {
        'C_W': 0.,
        'V_patch': 0.05,
        'Xdot_in': 100,
        'a_int': 6,
        'tspan': (0, 80),
        'T' : 8000
    }

    spc = {
        'X_emb_int': 19.42, 
        'eta_IA_0': 0.3333333333333333, 
        'eta_AS_0': 0.9, 
        'eta_AR_0': 0.95, 
        'Idot_max_rel': 12.256744759847304 ,
        'Idot_max_rel_emb': 12.256744759847304 ,
        'K_X': 500.0, 
        'kappa': 0.9, 
        'eta_SA': 0.9, 
        'k_M_0': np.float64(0.5039684199579493), 
        'S_p': 258.93333333333334, 
        'kD_j': 0.5, 
        'ED50_j': 1.0, 
        'beta_j': 2.0, 
        'pmoa': 'M', # PMoA wird in setup_modelfit überschrieben!
        'kD_h': 0.5, 
        'ED50_h': 2.0, 
        'beta_h': 1.0,
        'T_a' : 8000
        }
    
    # geschätzte DEB-Parameter aus der vorherigen Übung

    fitted_params =  { #parameter mit 25, aber T_a nicht sehr sinnvoll?
        'Idot_max_rel': np.float64(1.7293073035956965),
        'k_M_0': np.float64(0.1891084716041807), 
        'T_A': np.float64(-691.7697988262858)
        }
    
    spc.update(fitted_params)

    p = Params(glb = glb, spc = spc)

    return p

def define_loss(constants = None):

    def loss(
        prediction: pd.DataFrame,
        data: pd.DataFrame, 
        ) -> float: # returns a scalar value

        # Zusammenfassung von Vorhersagen und Daten in einen einzelnen Datensatz
        eval_df = pd.merge(
            prediction, 
            data, 
            on = ['t_day','T_cels','C_W'], 
            how = 'right', 
            suffixes = ['_predicted', '_observed']
            )
        
        # Berechnung der Loss-Funktion
        loss_S = logMSE(eval_df.S_predicted, eval_df.S_observed)
        
        return loss_S 
    
    return loss


def setup_modelfit(pmoa = 'G'):
    
    f = ModelFit()
    f.data = load_data()

    # Konstanten die während des Fittings genutzt werden
    
    constants = {
        'scale_factor_S' : np.max(f.data.loc[lambda df : df['C_W']==0].S)
    }
    
    # Skalierung der Daten

    f.data = f.data.assign( 
        S_scaled = lambda df : df.S / constants['scale_factor_S'] # Berechnung der skalierten Struktur
        )

    # Definition von Anfangswerten der Parameter

    f.defaultparams = define_defaultparams() # enthält alle Parameter, die für die Simulationen notwendig sind
    f.defaultparams.spc['pmoa'] = pmoa

    # enthält nur die Parameter, die kalibriert werden
    # was nicht in intugess ist, wird auf den wert in defaultparams fixiert 
    
    f.intguess = { 
        'kD_j' : 1.,
        #'ED50_j' : np.median(EXPOSURES),
        'ED50_j' : 2500,
        'beta_j' : 2.
        }

    f.simulator = define_simulator(f)
    f.loss = define_loss(constants)

    # define_objective_function kombiniert simulator und loss in eine einzelne Funktion
    # wenn sich eins von beiden ändert, muss auch define_ojective_function neu aufgerufen werden!
    f.define_objective_function()

    return f

def fit_model(pmoa = 'G'):

    f = setup_modelfit(pmoa) # generiere ModelFit-Instanz
    f.run_optimization() # führe Kalibrierung mittels lokaler Optimisierung durch

    print(f"Estimated parameter values: {f.p_opt}")

    # Simulation optimisierter Parameter

    p = deepcopy(f.defaultparams)
    p.spc.update(f.p_opt)  

    sim_opt = f.simulator(p.spc)

    # Visual predictive check

    fig, ax = plot_data(f.data)
    ax = plot_sim(ax, sim_opt)


    return f