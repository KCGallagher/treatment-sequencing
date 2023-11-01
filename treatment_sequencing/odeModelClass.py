# ====================================================================================
# Abstract model class
# ====================================================================================

import numpy as np
import scipy.integrate
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import contextlib

class ODEModel():
    def __init__(self, params, **kwargs):
        # Initialise parameters
        self.params = params
        self.state = ['V']
        self.results_df = None
        self.name = "ODEModel"

        # Configure the solver
        self.dt = kwargs.get('dt', 1e-3)  # Time resolution to return the model prediction on
        self.abs_err = kwargs.get('abs_err', 1.0e-8)  # Absolute error allowed for ODE solver
        self.rel_err = kwargs.get('rel_err', 1.0e-6)  # Relative error allowed for ODE solver
        self.solver_method = kwargs.get('method', 'DOP853')  # ODE solver used
        self.suppress_output = kwargs.get('suppress_output',
                                          False)  # If true, suppress output of ODE solver (including warning messages)
        self.success = False  # Indicate successful solution of the ODE system

    def __str__(self):
        return self.name
    
    def model_eqns(self, t, state):
        return NotImplementedError
    
    # =========================================================================================
    # Helper function to solve an ODE system
    def _solve_ode(self, state, t_max, t_min=0, **kwargs):
        self.solver_method = kwargs.get('method', self.solver_method)
        self.suppress_output = kwargs.get('suppress_output',
                                          self.suppress_output) 
        # If true, suppress output of ODE solver (including warning messages)
        
        # Solve
        times = np.arange(t_min, t_max, self.dt)
        state = self.initial_state
        
        ode_input_args = {'y0': state, 't_span': (times[0], times[-1] + self.dt),
                          't_eval': times,'method': self.solver_method,
                          'atol': self.abs_err, 'rtol': self.rel_err,
                          'max_step': kwargs.get('max_step', 1)}
        if self.suppress_output:
            with stdout_redirected():
                solution = scipy.integrate.solve_ivp(self.model_eqns, **ode_input_args)
        else:
            solution = scipy.integrate.solve_ivp(self.model_eqns, **ode_input_args)
        
        # Check that the solver converged
        encountered_problem = False
        if not solution.success or np.any(solution.y < 0):
            self.err_message = solution.message
            encountered_problem = True
            if not self.suppress_output: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            if not solution.success:
                if not self.suppress_output: print(self.err_message)
            else:
                if not self.suppress_output: print(
                    "Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver.")
                if not self.suppress_output: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            self.solution = solution
            
        self.success = True if not encountered_problem else False
        return pd.DataFrame({"Time": times, **dict(zip(self.state,solution.y))})

    # =========================================================================================
    # Function to simulate the model
    def simulate(self, t_max, t_min=0, **kwargs):
        # Allow configuring the solver at this point as well
        self.dt = float(kwargs.get('dt', self.dt))  # Time resolution to return the model prediction on
        self.abs_err = kwargs.get('abs_err', self.abs_err)  # Absolute error allowed for ODE solver
        self.rel_err = kwargs.get('rel_err', self.rel_err)  # Relative error allowed for ODE solver
        self.solver_method = kwargs.get('method', self.solver_method)  # ODE solver used
        self.success = False  # Indicate successful solution of the ODE system
        self.suppress_output = kwargs.get('suppress_output',
                                          self.suppress_output)  # If true, suppress output of ODE solver (including warning messages)

        # Solve
        self.initial_state = [self.params[var + "0"] for var in self.state]
        self.results_df = self._solve_ode(self.initial_state, t_max=t_max, t_min=t_min)

    # =========================================================================================
    # Function to plot the model predictions
    def plot(self, ax=None, vars_to_plot = ['V'], decorate_axes=True, apply_tight_layout=True, **kwargs):
        if ax is None: _, ax = plt.subplots(1, 1)
        model_df = pd.melt(self.results_df, id_vars=['Time'], value_vars=vars_to_plot)
        ax.plot(model_df.Time, model_df.value, **kwargs)

        # Format the plot
        ax.set_xlabel("Time (Days)" if decorate_axes else "")
        ax.set_ylabel("Tumor Size" if decorate_axes else "")
        ax.set_title(kwargs.get('title', ''))

        if apply_tight_layout:
            plt.tight_layout()
        if kwargs.get('saveFigB', False):
            plt.savefig(kwargs.get('outName', 'modelPrediction.png'), orientation='portrait', format='png')
            plt.close()


# ====================================================================================
# Functions used to suppress output from odeint
# Taken from: https://stackoverflow.com/questions/31681946/disable-warnings-originating-from-scipy
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied