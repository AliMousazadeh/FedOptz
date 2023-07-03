# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Starting code for the Federated Learning project. Some functions are explicitly left blank for students to fill in.

## Setup
#### Environment
If not working on CoLab, install environment with conda (preferred): 
```bash 
conda env create -f mldl23fl.yml
```

#### Datasets
The repository supports experiments on the following datasets:
1. **FEMNIST** (Federated Extended MNIST) from LEAF benchmark [1]
   - Task: image classification on 62 classes
   - 3,500 users
   - Instructions for download and preprocessing in ```data/femnist/``` 
2. Reduced **Federated IDDA** from FedDrive [2]
   - Task: semantic segmentation for autonomous driving
   - 24 users

## How to run
The ```main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of FedAvg experiment (**NB** training hyperparameters need to explicitly specified by the students):

- **FEMNIST** (Image Classification)
```bash
python load_main.py --dataset femnist --niid --model cnn --num_rounds 200 --num_epochs 1 --clients_per_round 5 --bs 4 --lr 0.001 --test_interval 10 --data_portion 25 --method fedoptz --alpha 0.2
```

## References
[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." Workshop on Federated Learning for Data Privacy and Confidentiality (2019). 

[2] Fantauzzo, Lidia, et al. "FedDrive: generalizing federated learning to semantic segmentation in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.
