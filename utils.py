import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ODEout(sim, lw = 2):
    """
    Plot the output of simout: All major state variable trajectories per exposure concentration.
    """

    # plot result of simulating chemical exposure
    if 'C_W' in sim.columns:

        fig, ax = plt.subplots(ncols = 4, nrows = 2, sharex = True, figsize = (10, 6))
        ax = np.ravel(ax)
        plt.tight_layout()
        sns.despine()

        for C_W in np.unique(sim.C_W): # iterate over concentrations

            df = sim.loc[sim['C_W'] == C_W]

            sns.lineplot(df, ax = ax[0], x = 't_day', y = 'S', linewidth = lw)
            sns.lineplot(df, ax = ax[1], x = 't_day', y = 'R', linewidth = lw)
            sns.lineplot(df, ax = ax[2], x = 't_day', y = 'X_emb', linewidth = lw, label = C_W)
            sns.lineplot(df, ax = ax[3], x = 't_day', y = 'X', linewidth = lw)
            sns.lineplot(df, ax = ax[4], x = 't_day', y = 'D_j', linewidth = lw)
            sns.lineplot(df, ax = ax[5], x = 't_day', y = 'D_h', linewidth = lw)
            sns.lineplot(df, ax = ax[6], x = 't_day', y = 'survival', linewidth = lw)

        ax[0].set(xlabel = "Time (d)", ylabel = "Structure (mugC)", xticks = (0,7,14,21))
        ax[1].set(xlabel = "Time (d)", ylabel = "Cumulative reproduction (mugC)")
        ax[2].set(xlabel = "Time (d)", ylabel = "Vitellus (mugC)")
        ax[3].set(xlabel = "Time (d)", ylabel = "Food resource (mugC)")
        ax[4].set(xlabel = "Time (d)", ylabel = "Sublethal damage (nmol L-1)")
        ax[5].set(xlabel = "Time (d)", ylabel = "Lethal damage (nmol L-1)")
        ax[6].set(xlabel = "Time (d)", ylabel = "Survival probability (-)", ylim = (0, 1.05))

        fig.delaxes(ax[7])
        ax[2].legend(title = r'$C_W\ (nmol\ L^{-1})$')
        
        return fig, ax
    
    # plot simulation results without chemical exposure
    else:
                
        statevars = ["S", "R"] # state variables to plot
        fig, ax = plt.subplots(ncols = len(statevars), figsize = (10,4)) # create subplots according to number of state variables

        for (i,y) in enumerate(statevars):
            sns.lineplot(
                sim, x = "t", y = y, # plotting state over time
                ax = ax[i] # plot on the ith subplot
                )

        sns.despine() # removes bounding box

        return fig,ax