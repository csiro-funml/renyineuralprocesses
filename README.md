# Rényi Neural Processes

## This repository accompanies the paper  <u>["Rényi Neural Processes"](https://arxiv.org/abs/2405.15991)
 
### <u>[Xuesong Wang](https://www.linkedin.com/in/xuesong-wang-7728a711a/)</u>, <u>[He Zhao](https://scholar.google.com/citations?user=pkn0NPsAAAAJ&hl=en)</u>, and <u>[Edwin V. Bonilla](https://scholar.google.com.au/citations?user=uDLRZQMAAAAJ&hl=en)</u> . ICML 2025 (**Oral, ~1 % of total submissions**)
---

## 1D Regression
---
### Training
```
python regression/main_gp.py --data_name=RBF --model_name=NP --mode=train --divergence=Renyi_0.7    
```
The config of hyperparameters of each model is saved in `regression/configs/gp`. If training for the first time, evaluation data will be generated and saved in `regression/evalsets/gp`. Model weights and logs are saved in `regression/results/{data_name}/{model_name}/`.

### Evaluation
```
python regression/main_gp.py --data_name=RBF --model_name=NP --mode=eval --divergence=Renyi_0.7    
```

### Plotting
```
python regression/main_gp.py --data_name=RBF --model_name=NP --mode=plot --divergence=Renyi_0.7    
```


Some codes are borrowed from https://github.com/tung-nd/TNP-pytorch , please refer to their documentation for more details for the dataset.



## Lotka Volterra & Hare Lynx datasets

---


### Simulation Data Generation (Lotka Volterra)
```
python3 data/lotka_volterra.py --filename=train --num_batches=10000 --trajectory_all=0

python3 data/lotka_volterra.py --filename=eval --num_batches=1000 --trajectory_all=0
```

The code will generate `datasets/lotka_volterra/train.tar` and `'datasets/lotka_volterra/eval.tar'


### Realworld Data Generation (Hare Lynx)
```
python3 data/hare_lynx.py
```

The code will download the dataset to `datasets/lotka_volterra/LynxHare.txt`


### Training
```
python regression/main_lotka_volterra.py --data_name=lotka_volterra --model_name=NP --mode=train --divergence=Renyi_0.7    
```

---

## Citation

Please cite us if you use this work:

_Xuesong Wang, He Zhao, Edwin V. Bonilla.
The Forty-Second International Conference on Machine Learning (ICML), 2025._

```bibtex
@article{wang2025rnp,
  title={R$\backslash$'enyi Neural Processes},
  author={Wang, Xuesong and Zhao, He and Bonilla, Edwin V},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```


## Acknowledgements
- Transformer neural processes: https://github.com/tung-nd/TNP-pytorch
- Lotka volterra: https://github.com/juho-lee/bnp
