# LION
This is the source code of the paper named **"Entropy and Autoencoder-Based Outlier Detection in Mixed-Type Network Traffic Data"** .


## Dataset 
- IDS17-Tu
- IDS17-We
- IDS12-Th
- IDS17-Fr1
- IDS17-Fr2
- IDS17-Fr3
- IDS17-Sum



## Dependencies
```
Python 3.6
Tensorflow == 1.12.0
pandas == 0.23.0
scikit-learn == 0.19.1
numpy == 1.14.3
```

## To run EAOD
1. run main.py for sample usage.  
2. Data set format: the name of categorical attributes should be named as "A1", "A2", ..., and the numerical ones are "B1", "B2", ...  
3. The input path can be an individual data set or just a folder.  
