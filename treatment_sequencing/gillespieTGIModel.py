# ====================================================================================
# Simulation class for the TGI model using Gillespie Tau-Leaping
# The tumor size is simulated discretely, while evolution speed gamma remains continuous
# ====================================================================================


import numpy as np
import pandas as pd

from .odeModelClass import ODEModel

class GillespieTGIModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GillespieTGIModel"
        self.state = ['V', 'gamma']
        
        
    def _calc_propensities(self, state):
        # Returns un-normalised values
        doubling_rate = self.params['lambda'] / np.log(2) 
        death_rate = state[2] / np.log(2)  # Final state value is gamma
        return np.array([doubling_rate, death_rate]) * state[1]


    def simulate(self, t_max, t_min=0, **kwargs):
        self.initial_state = [self.params[var + "0"] for var in self.state]
        state = np.zeros(len(self.initial_state) + 1)
        state[0] = t_min
        state[1:] = self.initial_state
        
        initial_data = {"Time": t_min, **dict(zip(self.state, self.initial_state))}
        self.results_df = pd.DataFrame(data=initial_data, index=[0])

        while state[0] < t_max:
            propensity_values = self._calc_propensities(state)
            total_rate = sum(propensity_values)
    
            time_step = np.log(1 / np.random.rand()) / total_rate
            state[0] += time_step
            
            # Update gamma value according to ODE
            state[2] = self.params['gamma0'] * np.exp(-self.params['epsilon'] * state[0])

            normal_prop = propensity_values / total_rate
            if np.random.rand() < normal_prop[0]:
                state[1] += 1  # Birth process
            else:
                state[1] -= 1  # Death process
            
            temp_df = pd.DataFrame([state[0], *state[1:]], index=["Time", *self.state])
            self.results_df = pd.concat([self.results_df, temp_df.transpose()])
            
            if state[1] == 0:
                break
            
        self.results_df.reset_index(inplace=True, drop=True)
