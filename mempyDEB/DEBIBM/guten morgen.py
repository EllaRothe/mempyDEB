from collections import namedtuple

Params = namedtuple("Params", "glb spc")

defaultparams_DEBIBM = Params(
    { # global parameters
    # Simulation parameters
    'tspan' : (0,365),
    'tres' : 24, # temporal resolution (timesteps per day)
    'N_0' : 10, # initial population size
    'data_collection_interval' : 3.5, # intervals at which data is collected (days)
    'collect_agent_data' : True, # whether to collect agent-level data or not
    'replicates' : 3, # run replicates of the simulation

    # Environmental parameters
    'V_patch':  0.5, # volume of a single patch
    'Xdot_in': 1250, # resource input rate
    'kX_out' : 0.1, # daily resource outflow rate
    'C_W' : 0. # chemical stressor concentration
    },
    { # animal parameters
    # DEB parameters
    'cv' : 0.1, # individual variability in DEB parameters, given as coefficient of variation 
    'Idot_max_rel_mean': 5, # maximum specific ingestion rate; theoretical population average
    'eta_IA_0': 0.5, # assimilation efficiency
    'K_X': 0.5e3, # half-saturation constant for resource uptake
    'kappa': 0.9, # somatic allocation fraction
    'eta_AS_0' : 0.9, # growth effieciency (transformation from assimilates to structure)
    'eta_SA' : 0.9, # structural mobilization efficiency (transformation from structure to asismilates)
    'eta_AR_0': 0.95, # reproduction efficiency
    'k_M_0': 0.55, # somatic maintenance rate constant; theoretical population average
    'S_p_mean': 9, # structural mass at puberty; theoretical population average
    'X_emb_int_mean': 0.675, # initial mass of the vitellus (~~ dry mass of an egg); theoretical population average
    'tau_R' : 2.5, # reproduction period (days)

    # starvation parameters
    'S_rel_crit' : 0.5, # critical amount of structure, relative to historical maximum structure of the individual
    'h_starve' : 0.29, # hazard rate applied when loss of structure is critical (corresponds to 75% daily survival probability)

    # aging parameters
    'a_max_mean' : 60, # average maximum lifespan
    'a_max_cv' : 0.1, # coefficient of variation in maximum lifespans

    # sublethal TKTD parameters 
    'kD_j' : .5, # dominant rate constant
    'ED50_j' : 1., # median effective damage
    'beta_j' : 1., # DRC slope
    'pmoa' : 'R', # physiological mode of action

    # GUTS parameters
    'kD_h' : .1, # dominant rate constant
    'ED50_h' : 2, # median effective damage
    'beta_h' : 1. # DRC slope
    }
)