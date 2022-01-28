## mentored [![DOI](https://zenodo.org/badge/451429201.svg)](https://zenodo.org/badge/latestdoi/451429201)

mentored (MEtadata aNnotaTiOn pREDiction) is a BiLSTM neural network trained to predict UMLS Codes to a given metadata description. The network consists of three layers: embedding layer, BiLSTM with 2 cells and a linear classifier.  

![Network Archicture](doc/network.png)

# results

| Model                            | t1           | t3           | t5           |     t10      |
|----------------------------------|--------------|--------------|--------------|--------------|
|     2BiLSTM   w/ Augmentation    |     67.27    |     72.85    |     74.23    |     75.63    |

|     Phrase                                        |     1. Prediction                            |     2. Prediction                                     |     3. Prediction                              |
|---------------------------------------------------|----------------------------------------------|-------------------------------------------------------|------------------------------------------------|
|     Name des Patienten (name of the patient)      |     Patient surname (25.04%)                 |     Medication name (18.32%)                          |     Patient forename (17.45%)                  |
|     Krebs der Niere     (cancer of the kidndy)    |     Kidney   cancer (24.84%)                 |     Subject   Diary (17.16%)                          |     Malignant   neoplasm of kidney (16.12%)    |
|     Krebs der Leber (cancer of the liver)         |     Malignant Placental Neoplasm (21.20%)    |     Secondary malignant neoplasm of liver (18.50%)    |     Liver reconstruction (17.72 %)             |
|     Blut (blood)                                  |     Blood   (19.38%)                         |     Blood in Urine (17.81%)                           |     Coagulation Process (17.21%)               |

# usage 

You need the data files in order to run the training. Two files are needed: MDM metadata and the corresponding MeSH files. 
Both files need to have the following shapes: "CODE";"PHRASE" e.g. "C0027989;newspapers"

Put the files in the data directory, change to path in the file "start_training" and run! 
