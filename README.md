Sepsis Prediction Methodology
==============================
This repository contains a rewritten version of the code for our submission (Team Name: Can I get your Signature?) to the PhysioNet 2019 challenge. 


Getting Started
---------------
Once the repo has been cloned locally, setup a python environment with ``python==3.7`` and run ``pip install -r requirements.txt``.

You then need to add the project root directory to your virtualenv python's path. This can be done by adding the location of the root folder into the site_packages.pth directory.
(See Arjen P. De Vries answer here: https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv/47184788#47184788).

Create a folder `/data/raw`, the `data/` folder should be made a symlink if you wish to store the large data files elsewhere. 

Run the following:
1. ``python src/data/get_data/download.py`` To download the raw .psv files to `/data/raw`
2. ``python src/data/get_data/convert_data.py`` To convert the downloaded data into a pandas dataframe (for easy analysis) and a TimeSeriesDataset (for fast operations).
    
You are then ready to go! Check `/notebooks/examples/prediction.ipynb` for an intro to the basic prediction methods and the functions used to generate the features. Then either follow the example and use a notebook to build your own models, or do something similar to that seen in ``src/model/examples/train_{MODEL_TYPE}.py``. 

More functionality will be added soon!

Repository Structure
--------------------
```
PhysioNet2019_Sepsis_Prediction/
├── configs/
├── data/
│   ├── experiment/
│   ├── processed/
│   │   └── labels/
│   └── raw/
│       ├── training_setA/
│       └── training_setB/
├── notebooks/
│   └── examples/
├── results/
│   ├── baseline_1/
│   ├── baseline_1_full/
│   ├── baseline_2-1/
│   ├── baseline_2-1_full/
│   ├── baseline_2-2/
│   │   └── saits/
│   ├── baseline_3-1/
│   └── baseline_3-2/
└── src/
    ├── data/
    │   └── get_data/
    ├── experiments/
    ├── external/
    │   └── example_solution/
    ├── features/
    │   └── signatures/
    ├── model/
    │   └── examples/
    ├── omni/
    └── visualization/
```