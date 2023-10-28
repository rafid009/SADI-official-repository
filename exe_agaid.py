from models.main_model import SADI_Agaid
from datasets.dataset_agaid import get_dataloader
from utils.utils import *
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS, BRITS
from datasets.process_data import *
import pickle
from config_ablation import partial_bm_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config.config_AgAid import config

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(10, 100)

data_file = 'data/AgAid/ColdHardiness_Grape_Merlot_2.csv'

train_loader, valid_loader, mean, std = get_dataloader(
    seed=seed,
    filename=data_file,
    batch_size=config["train"]["batch_size"],
    missing_ratio=0.2,
    season_idx=[32, 33]
)

print(config)

model_SADI = SADI_Agaid(config, device, is_simple=False).to(device)
filename = f'model_sadi_agaid.pth'
model_folder = "saved_model_agaid"
# train(
#     model_SADI,
#     config["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{filename}"
# )
nsample = 50
model_SADI.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"SADI params: {get_num_params(model_SADI)}")

models = {
    'SADI': model_SADI,
}
features_test = [1,3,5,7,9,11]
 
partial_bm_config['length_range'] = (30,30)
mse_folder = f"results_agaid_{partial_bm_config['features']}/metric"
data_folder = f"results_agaid_{partial_bm_config['features']}/data"

for i in partial_bm_config:
    partial_bm_config['features'] = i
    evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, test_indices=[32,33], mean=mean, std=std, partial_bm_config=partial_bm_config)


lengths = [50, 100, 150]

print(f"\nBlackout:\n")
for l in lengths:
    print(f"length = {l}")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, length=l, test_indices=[32,33], mean=mean, std=std)

lengths = [50, 100, 150]
for l in lengths:
    print(f"\nForecasting = {l}")
    evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, length=l, forecasting=True, test_indices=[32,33], mean=mean, std=std)

miss_ratios = [0.2, 0.3, 0.5]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})\n")
    evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, missing_ratio=ratio, random_trial=True, test_indices=[32,33], mean=mean, std=std)
