
# Machine Learning Engineer Capstone Project
### Project Overview 

This capstone project uses the data from Arvato project. The situation of Arvato is like other business that have to grow by expanding customer based to broader audiences. However, each customer acquisition requires some effort either financial or non-financial. In order to find potential customers while minimizing the cost, we should analyze populations demographics and other attributes against current customers characteristics therefore, we have some idea on what the prospect customers would look like. 

We can use the findings to build a predictive model that can give us the probability of if the customer reach-out email would be effective to each customer. 


### Customers and Populations Clustering
You can find this in the notebook named: `Population Clustering.ipynb`
#### Methodology
##### Data Preprocessing

There are several missing values and some categorical attributes, I clean it by

1.  Attribute Dropping  
2.  Missing Value Substitution  
3.  Data Imputation  
4.  Data Scaling  
##### Dimensionality Reduction
Since the input data composed of >300 features, to accommodate the clustering algorithm, I applied the Principal Component Analysis (PCA) to reduce the data dimensions to the handleable scope. The question that would come up is how many dimensions that we should pick.
I use the proportion of explained variance to identify number of target reduced dimensions. In this case, I found that ***if we use number of dimensions at 125, they can explain ~80% of data variance*** which is decent for further usage.
##### Data Clustering
After applying the dimensionality reduction, the data we obtained have 125 attributes. They are injected into KMean clustering. Another question coming up is how many clusters (K) we should use. One of the approaches is elbow method to identify how many clusters that give diminishing reduction in sum squared intra-cluster distance.
##### Findings
From this clusting we can see that customers tend to have degree which is the highest different. Another difference between population and customer is that customer tend to have lower portion of attributes that describe about people with >= 50 years old.

***This could roughly indicate that customers and the prospective customers are likely to be ones with younger and educated people.***

### Customer Response Prediction
You can it in the notebook named: `XGBoost - Customer Response Prediction.ipynb`
#### Methodology
This model is mented to be run on Amazon Sagemaker.
XGBoost Model Training and Hyperparameter Tuning

XGBoost model is chosen due to its renounced. performance in handling with tabular data with high dimensional features. It also uses less resources to train compared to many other model architectures.

##### Imbalanced Data

According to the training dataset, positive data is only 1.2% of total dataset. Therefore, to train the XGBoost, we have to give more weightage to positive data by defining scale_pos_weight = 80 to make positive data comparable to the negative data (the negative data is ~80 times more prevalent than the positive data).

##### Initial Hyperparameter Definition

The following shows the initial hyperparameters settings on SageMaker XGBoost  model.

    xgb.set_hyperparameters(max_depth=3,
						    eta=0.05,
						    gamma=5.5,
						    min_child_weight=3,
						    subsample=0.8,
						    objective='binary:logistic',
						    early_stopping_rounds=20,
						    num_round=200,
						    scale_pos_weight=80)

The key thing here is the objective of the model training, I set the objective as ‘binary:logistic’ as it is suitable with our measurement metric which is AUC ROC.

##### Hyperparameter Definition for Model Refinement

Since we do not know what hyperparameters can deliver us the best model performance, I use the hyperparameter tuner model in SageMaker package to identify best model configurations.

The following shows the definition of hyperparameters tuning job.

    HyperparameterTuner(estimator = xgb,
			    objective_metric_name = 'validation:auc',
			    objective_type = 'Maximize',
			    max_jobs = 20,
			    max_parallel_jobs = 3,
			    hyperparameter_ranges = {
				    'max_depth': IntegerParameter(3, 12),
				    'eta' :  ContinuousParameter(0.05, 0.5),
				    'min_child_weight': IntegerParameter(2, 8),
				    'subsample': ContinuousParameter(0.5, 0.9),
				    'gamma': ContinuousParameter(0, 10),
			    },
			    strategy='Bayesian')

A few important settings are.

-   Strategy: In this case we use Bayesian optimization which should be able to find the better model faster than ordinary grid search and random search.
    
-   Hyperparameter Ranges: Definitions of ranges of the XGBoost hyperparameters.
    
-   Objective Metric: We use the AUC score of the validation dataset to identify which model deliver best performance.
    
-   Max Jobs: Number of model training jobs.
    

##### Trained Model Results

After using the hyperparameter tuning, we can identify the suitable hyperparameters that can be used for test data inference.

The results of best-found model give us the AUC score at 0.77 on validation dataset.
#### Model Results
By submitting ths model to the Kaggle competition, the performance of AUC score is at `0.79430`
