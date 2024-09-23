# DATATHON2024 Model Training Code

### findFirstFilledYear.py
The file ***findFirstFilledYear.py*** was written to determine the years in which the columns in the dataset were first filled. The years here are taken from the values in the dataset that correspond to the year of application. By analyzing when the data started to fill in each column, this script provides important information about how the model evolves over time.

### missingValuesCheck.py
The ***missingValuesCheck.py*** file shows the amount of missing (empty) data in columns in the dataset. The purpose of this check is to optimize the dataset by identifying columns that do not have an impact on the 'Assessment Score' during the modeling process or that reduce data quality, and by excluding unnecessary columns from the model. In this way, we aim to improve the performance of the model and reduce the computational burden.

### UpdatedTry8.py
***UpdatedTry8.py*** performed significantly better than the other subbisions for the first time when I used it with the CatBoosting algorithm when I paid more attention to data processing, unlike my other experiments. This was achieved by paying particular attention to data cleaning, processing missing values and selecting appropriate hyperparameters. As a result, I was able to create a significant performance difference from previous submissions.

### UpdatedTry10.py
***UpdatedTry10.py*** is a slightly improved version of UpdatedTry8.py.Unfortunately, due to time constraints while writing and working on this code, I was not able to spend enough time on the training of the model and the susbmission process. However, it is a version that I believe I could have achieved much higher performance if I had been able to work on it more.
