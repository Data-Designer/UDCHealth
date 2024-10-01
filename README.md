<h1 align="center"> Unveiling Discrete Clues: Superior Healthcare Predictions for Rare Diseases </h1>


## About Our Work

Update: 2024/09/26: We have created a repository for the paper titled *Unveiling Discrete Clues: Superior Healthcare Predictions for Rare Diseases*, which has been submitted to the *WWW 2025*. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20240926141140345](https://s2.loli.net/2024/09/26/qQiB2ObLJzvRCjy.png)

![image-20240926141209183](https://s2.loli.net/2024/09/26/7J56M3YEdcGgBm9.png)

## Requirements

- openai==1.3.5
- torch==1.13.1+cu117
- dgl==1.1.2
- pyhealth==1.1.4
- seaborn==0.13.0

## Data Sets

Owing to the copyright stipulations associated with the dataset, we are unable to provide direct upload access. However, it can be readily obtained by downloading directly from the official website: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/),[eICU](https://eicu-crd.mit.edu/). 

The structure of the data set should be like,

```powershell
data
|_ DIAG
|  |_ MIII
|  |_ _processed
|  |_ _ _datasets_pre_stand.pkl
|  |_ _ rare.pkl
|  |_ MIV
|  |_ _ _datasets_pre_stand.pkl
|_ REC
|_ _MIII
|  |_ _processed
|  |_ _ _datasets_pre_stand.pkl
|  |_ _ rare_patient.pkl
```

## RUN

```powershell
# run the main file
# change config.py
config = vars(UDCDIAGConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}

# please set pretrain=True (first, sencod stage) 
# please set tuning=True (third stage)
python main.py
```

## Acknowledge & Contact

We will update the contact information and corresponding issue links after the review process is completed.