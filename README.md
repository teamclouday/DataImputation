# Data Imputation Research  

In the process of machine learning, input data always plays an essential role in the model training  
However, real world data is usually not perfect. Very often part of the dataset is missing.  
This repo is created to research on different methods of data replacing and guessing, and their according effect to several well-known machine learning algorithms.  

Specifically, this research is trying to find out how the missing data would make the output of machine learning algorithms "biased".  

------

### Main Variables  
1. The methods of generating missing data entries  
2. The methods of guessing a missing data entry  
3. Different machine learning algorithms to test on  

------

### Datasets
1. [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/Iris) (UCI) (No Longer Used)  
2. [Bank Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (UCI)  
3. [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) (UCI)  
4. [Compas Dataset](https://github.com/propublica/compas-analysis/)  
5. [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (UCI) (No Longer Used)  
6. [Drug Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29) (UCI) (No Longer Used)  
7. [Titanic Dataset](https://www.kaggle.com/c/titanic) (Kaggle) (Kaggle account required)  
8. [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (UCI)  
9. [Communities and Crime Dataset](http://archive.ics.uci.edu/ml/datasets/communities+and+crime) (UCI)  
10. [Recidivism in juvenile justice](http://cejfe.gencat.cat/en/recerca/opendata/jjuvenil/reincidencia-justicia-menors/index.html) (No Longer Used)  

------

### References  
1. [IBM AIF360](https://github.com/Trusted-AI/AIF360)  
2. [Missing-data imputation](http://www.stat.columbia.edu/~gelman/arm/missing.pdf)  
3. [Multiple Imputation in Stata](https://stats.idre.ucla.edu/stata/seminars/mi_in_stata_pt1_new/)  
4. [COMPAS Recidivism Risk Score Data and Analysis](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)  
5. [Responsibily](https://docs.responsibly.ai/index.html)  
6. [Fairness Measures](http://www.fairness-measures.org/)  
7. More in `research notes.ipynb`  

------

### Note  
Due the multiprocessing nature of Python3, scripts involving multiprocessing cannot be run on Windows.  