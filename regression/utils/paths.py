import os

server_folder = 'renyi_neural_process/regression'
ROOT = '' if not os.path.exists(server_folder) else '/renyi_neural_process/regression'
SAVE_ROOT = '' if not os.path.exists(server_folder) else '/renyi_neural_process/regression'
evalsets_path = os.path.join(SAVE_ROOT, 'evalsets')
datasets_path = os.path.join(SAVE_ROOT, 'datasets')
results_path = os.path.join(SAVE_ROOT, 'results')
config_path = os.path.join(ROOT, 'configs')
