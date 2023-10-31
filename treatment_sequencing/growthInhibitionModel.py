# ====================================================================================
# Class to simulate simple tumor growth inhibition model
# ====================================================================================

import numpy as np

from .odeModelClass import ODEModel

class GrowthInhibitionModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GrowthInhibitionModel"
        self.state = ['V', 'gamma']

    # The governing equations
    def model_eqns(self, t, state):
        V, gamma = state
        state_change = np.zeros_like(state)
        state_change[0] = self.params['lambda'] * V - gamma * V
        state_change[1] = - self.params['epsilon'] * gamma
        return (state_change)
