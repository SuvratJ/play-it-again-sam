from src.tfp.offline_experiments.dataset import prepare_dataset
from src.tfp.offline_experiments.fine_tune_hill_climbing import _save as _save_hc
from src.tfp.offline_experiments.fine_tune_hill_climbing import plot as plot_hc
from src.tfp.algorithms import *
from src.tfp.offline_experiments.performance import _save as _save_performance
from src.tfp.offline_experiments.performance import plot as plot_performance
from src.tfp.offline_experiments.timing import _save as _save_timing
from src.tfp.offline_experiments.timing import timing_plot

prepare_dataset("main", 42)
prepare_dataset("side", 24)

# fine tune hill-climbing
_save_hc()
plot_hc()

# performance
algos = [optimal, hill_climbing, greedy, ]
_save_performance(algos)
plot_performance(algos, np.average, legend=False, x_axis_label='$|I|$', y_axis_label='$score$', title="(a) Average")
plot_performance(algos, np.std, legend=False, x_axis_label='$|I|$', title="(b) Standard deviation")
plot_performance(algos, max, legend=False, x_axis_label='$|I|$', title="(c) Maximum")
plot_performance(algos, min, legend=True, x_axis_label='$|I|$', title="(d) Minimum")

# timing
_save_timing(algos)
timing_plot(algos)
