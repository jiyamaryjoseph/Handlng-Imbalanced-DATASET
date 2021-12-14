# Handlng-Imbalanced-Dataset.(such as fraud event, payment default, spam)
Here I am using many technique for handling imbalanced dataset.


## Introduction
When working with an imbalanced classification problem, we are usually more concerned with correctly predicting the minority class (also called the positive class), e.g., fraud event, payment default, spam, etc. In other words, it is more critical for us to reduce the number of false negatives (e.g., classifying a fraudulent transaction as non-fraudulent) than false positives (e.g., classifying a normal email as spam).


If you have spent some time in machine learning and data science, you would have definitely come across imbalanced class distribution. This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes.

This problem is predominant in scenarios where anomaly detection is crucial like electricity pilferage, fraudulent transactions in banks, identification of rare diseases, etc. In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.

This happens because Machine Learning Algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.

This guide describes various approaches for solving such class imbalance problems using various sampling techniques. We also weigh each technique for its pros and cons. Finally, I reveal an approach using which you can create a balanced class distribution and apply ensemble learning technique designed especially for this purpose.

 

Table of Content
Challenges faced with Imbalanced datasets
Approach to handling Imbalanced Datasets
Illustrative Example
Conclusion


1. Challenges faced with imbalanced data
One of the main challenges faced by the utility industry today is electricity theft. Electricity theft is the third largest form of theft worldwide. Utility companies are increasingly turning towards advanced analytics and machine learning algorithms to identify consumption patterns that indicate theft.

However, one of the biggest stumbling blocks is the humongous data and its distribution. Fraudulent transactions are significantly lower than normal healthy transactions i.e. accounting it to around 1-2 % of the total number of observations. The ask is to improve identification of the rare minority class as opposed to achieving higher overall accuracy.

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.

Example of imbalanced data
Let’s understand this with the help of an example.

- Ex: In an utilities fraud detection data set you have the following data:

Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate= 2 %

The main question faced during data analysis is – How to get a balanced dataset by getting a decent number of samples for these anomalies given the rare occurrence for some them?

2.1 Data Level approach: Resampling Techniques
Dealing with imbalanced datasets entails strategies such as improving classification algorithms or balancing classes in the training data (data preprocessing) before providing the data as input to the machine learning algorithm. The later technique is preferred as it has wider application.

The main objective of balancing classes is to either increasing the frequency of the minority class or decreasing the frequency of the majority class. This is done in order to obtain approximately the same number of instances for both the classes. Let us look at a few resampling techniques:

 

2.1.1  Random Under-Sampling
Random Undersampling aims to balance class distribution by randomly eliminating majority class examples.

This is done until the majority and minority class instances are balanced out.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are taking 10 % samples without replacement from Non Fraud instances.  And combining them with Fraud instances.

Non Fraudulent Observations after random under sampling = 10 % of 980 =98

Total Observations after combining them with Fraudulent observations = 20+98=118
Event Rate for the new dataset after under sampling = 20/118 = 17%

- Advantages
It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.
- Disadvantages
It can discard potentially useful information which could be important for building rule classifiers.
The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.

Introduction
If you have spent some time in machine learning and data science, you would have definitely come across imbalanced class distribution. This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes.

This problem is predominant in scenarios where anomaly detection is crucial like electricity pilferage, fraudulent transactions in banks, identification of rare diseases, etc. In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.

This happens because Machine Learning Algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.

This guide describes various approaches for solving such class imbalance problems using various sampling techniques. We also weigh each technique for its pros and cons. Finally, I reveal an approach using which you can create a balanced class distribution and apply ensemble learning technique designed especially for this purpose.

 

Table of Content
Challenges faced with Imbalanced datasets
Approach to handling Imbalanced Datasets
Illustrative Example
Conclusion


1. Challenges faced with imbalanced data
One of the main challenges faced by the utility industry today is electricity theft. Electricity theft is the third largest form of theft worldwide. Utility companies are increasingly turning towards advanced analytics and machine learning algorithms to identify consumption patterns that indicate theft.

However, one of the biggest stumbling blocks is the humongous data and its distribution. Fraudulent transactions are significantly lower than normal healthy transactions i.e. accounting it to around 1-2 % of the total number of observations. The ask is to improve identification of the rare minority class as opposed to achieving higher overall accuracy.

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.

Example of imbalanced data
Let’s understand this with the help of an example.

Ex: In an utilities fraud detection data set you have the following data:

Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate= 2 %

The main question faced during data analysis is – How to get a balanced dataset by getting a decent number of samples for these anomalies given the rare occurrence for some them?

Challenges with standard Machine learning techniques
The conventional model evaluation methods do not accurately measure model performance when faced with imbalanced datasets.

Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

Evaluation of a classification algorithm performance is measured by the Confusion Matrix which contains information about the actual and the predicted class.


Accuracy of a model = (TP+TN) / (TP+FN+FP+TN)

However, while working in an imbalanced domain accuracy is not an appropriate measure to evaluate model performance. For eg: A classifier which achieves an accuracy of 98 % with an event rate of 2 % is not accurate, if it classifies all instances as the majority class. And eliminates the 2 % minority class observations as noise.

Examples of imbalanced data
Thus, to sum it up, while trying to resolve specific business challenges with imbalanced data sets, the classifiers produced by standard machine learning algorithms might not give accurate results. Apart from fraudulent transactions, other examples of a common business problem with imbalanced dataset are:

Datasets to identify customer churn where a vast majority of customers will continue using the service. Specifically, Telecommunication companies where Churn Rate is lower than 2 %.
Data sets to identify rare diseases in medical diagnostics etc.
Natural Disaster like Earthquakes
Dataset used
In this article, we will illustrate the various techniques to train a model to perform well against highly imbalanced datasets. And accurately predict rare events using the following fraud detection dataset:

Total Observations = 1000

Fraudulent   Observations =20

Non-Fraudulent Observations = 980

Event Rate= 2 %

Fraud Indicator = 0 for Non-Fraud Instances

Fraud Indicator = 1 for Fraud

 

2. Approach to handling Imbalanced Data
2.1 Data Level approach: Resampling Techniques
Dealing with imbalanced datasets entails strategies such as improving classification algorithms or balancing classes in the training data (data preprocessing) before providing the data as input to the machine learning algorithm. The later technique is preferred as it has wider application.

The main objective of balancing classes is to either increasing the frequency of the minority class or decreasing the frequency of the majority class. This is done in order to obtain approximately the same number of instances for both the classes. Let us look at a few resampling techniques:

 

2.1.1  Random Under-Sampling
Random Undersampling aims to balance class distribution by randomly eliminating majority class examples.  This is done until the majority and minority class instances are balanced out.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are taking 10 % samples without replacement from Non Fraud instances.  And combining them with Fraud instances.

Non Fraudulent Observations after random under sampling = 10 % of 980 =98

Total Observations after combining them with Fraudulent observations = 20+98=118

Event Rate for the new dataset after under sampling = 20/118 = 17%

 

Advantages
It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.
Disadvantages
It can discard potentially useful information which could be important for building rule classifiers.
The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.
 

2.1.2  Random Over-Sampling
Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are replicating 20 fraud observations   20 times.

Non Fraudulent Observations =980

Fraudulent Observations after replicating the minority class observations= 400

Total Observations in the new data set after oversampling=1380

Event Rate for the new data set after under sampling= 400/1380 = 29 %

Advantages
Unlike under sampling this method leads to no information loss.
Outperforms under sampling
Disadvantages
It increases the likelihood of overfitting since it replicates the minority class events.

Introduction
If you have spent some time in machine learning and data science, you would have definitely come across imbalanced class distribution. This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes.

This problem is predominant in scenarios where anomaly detection is crucial like electricity pilferage, fraudulent transactions in banks, identification of rare diseases, etc. In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.

This happens because Machine Learning Algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.

This guide describes various approaches for solving such class imbalance problems using various sampling techniques. We also weigh each technique for its pros and cons. Finally, I reveal an approach using which you can create a balanced class distribution and apply ensemble learning technique designed especially for this purpose.

 

Table of Content
Challenges faced with Imbalanced datasets
Approach to handling Imbalanced Datasets
Illustrative Example
Conclusion


1. Challenges faced with imbalanced data
One of the main challenges faced by the utility industry today is electricity theft. Electricity theft is the third largest form of theft worldwide. Utility companies are increasingly turning towards advanced analytics and machine learning algorithms to identify consumption patterns that indicate theft.

However, one of the biggest stumbling blocks is the humongous data and its distribution. Fraudulent transactions are significantly lower than normal healthy transactions i.e. accounting it to around 1-2 % of the total number of observations. The ask is to improve identification of the rare minority class as opposed to achieving higher overall accuracy.

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.

Example of imbalanced data
Let’s understand this with the help of an example.

Ex: In an utilities fraud detection data set you have the following data:

Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate= 2 %

The main question faced during data analysis is – How to get a balanced dataset by getting a decent number of samples for these anomalies given the rare occurrence for some them?

Challenges with standard Machine learning techniques
The conventional model evaluation methods do not accurately measure model performance when faced with imbalanced datasets.

Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

Evaluation of a classification algorithm performance is measured by the Confusion Matrix which contains information about the actual and the predicted class.


Accuracy of a model = (TP+TN) / (TP+FN+FP+TN)

However, while working in an imbalanced domain accuracy is not an appropriate measure to evaluate model performance. For eg: A classifier which achieves an accuracy of 98 % with an event rate of 2 % is not accurate, if it classifies all instances as the majority class. And eliminates the 2 % minority class observations as noise.

Examples of imbalanced data
Thus, to sum it up, while trying to resolve specific business challenges with imbalanced data sets, the classifiers produced by standard machine learning algorithms might not give accurate results. Apart from fraudulent transactions, other examples of a common business problem with imbalanced dataset are:

Datasets to identify customer churn where a vast majority of customers will continue using the service. Specifically, Telecommunication companies where Churn Rate is lower than 2 %.
Data sets to identify rare diseases in medical diagnostics etc.
Natural Disaster like Earthquakes
Dataset used
In this article, we will illustrate the various techniques to train a model to perform well against highly imbalanced datasets. And accurately predict rare events using the following fraud detection dataset:

Total Observations = 1000

Fraudulent   Observations =20

Non-Fraudulent Observations = 980

Event Rate= 2 %

Fraud Indicator = 0 for Non-Fraud Instances

Fraud Indicator = 1 for Fraud

 

2. Approach to handling Imbalanced Data
2.1 Data Level approach: Resampling Techniques
Dealing with imbalanced datasets entails strategies such as improving classification algorithms or balancing classes in the training data (data preprocessing) before providing the data as input to the machine learning algorithm. The later technique is preferred as it has wider application.

The main objective of balancing classes is to either increasing the frequency of the minority class or decreasing the frequency of the majority class. This is done in order to obtain approximately the same number of instances for both the classes. Let us look at a few resampling techniques:

 

2.1.1  Random Under-Sampling
Random Undersampling aims to balance class distribution by randomly eliminating majority class examples.  This is done until the majority and minority class instances are balanced out.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are taking 10 % samples without replacement from Non Fraud instances.  And combining them with Fraud instances.

Non Fraudulent Observations after random under sampling = 10 % of 980 =98

Total Observations after combining them with Fraudulent observations = 20+98=118

Event Rate for the new dataset after under sampling = 20/118 = 17%

 

Advantages
It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.
Disadvantages
It can discard potentially useful information which could be important for building rule classifiers.
The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.
 

2.1.2  Random Over-Sampling
Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are replicating 20 fraud observations   20 times.

Non Fraudulent Observations =980

Fraudulent Observations after replicating the minority class observations= 400

Total Observations in the new data set after oversampling=1380

Event Rate for the new data set after under sampling= 400/1380 = 29 %

Advantages
Unlike under sampling this method leads to no information loss.
Outperforms under sampling
Disadvantages
It increases the likelihood of overfitting since it replicates the minority class events.
 

2.1.3  Cluster-Based Over Sampling
In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class have an equal number of instances and all classes have the same size.  

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

Majority Class Clusters
Cluster 1: 150 Observations
Cluster 2: 120 Observations
Cluster 3: 230 observations
Cluster 4: 200 observations
Cluster 5: 150 observations
Cluster 6: 130 observations
Minority  Class Clusters
Cluster 1: 8 Observations
Cluster 2: 12 Observations
 

After oversampling of each cluster, all clusters of the same class contain the same number of observations.

Majority Class Clusters
Cluster 1: 170 Observations
Cluster 2: 170 Observations
Cluster 3: 170 observations
Cluster 4: 170   observations
Cluster 5: 170   observations
Cluster 6: 170   observations
Minority   Class Clusters
Cluster 1: 250 Observations
Cluster 2: 250 Observations
Event Rate post cluster based oversampling sampling = 500/ (1020+500) = 33 %
- Advantages
This clustering technique helps overcome the challenge between class imbalance. Where the number of examples representing positive class differs from the number of examples representing a negative class.
Also, overcome challenges within class imbalance, where a class is composed of different sub clusters. And each sub cluster does not contain the same number of examples.
- Disadvantages
The main drawback of this algorithm, like most oversampling techniques is the possibility of over-fitting the training data.

2.1.4  Informed Over Sampling: Synthetic Minority Over-sampling Technique for imbalanced data(SMOTE)
This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.
Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate = 2 %

A sample of 15 instances is taken from the minority class and similar synthetic instances are generated 20 times

Post generation of synthetic instances, the following data set is created

Minority Class (Fraudulent Observations) = 300

Majority Class (Non-Fraudulent Observations) = 980

Event rate= 300/1280 = 23.4 %

 

- Advantages
Mitigates the problem of overfitting caused by random oversampling as synthetic examples are generated rather than replication of instances
No loss of useful information
- Disadvantages
While generating synthetic examples SMOTE does not take into consideration neighboring examples from other classes. This can result in increase in overlapping of classes and can introduce additional noise
SMOTE is not very effective for high dimensional data.

Introduction
If you have spent some time in machine learning and data science, you would have definitely come across imbalanced class distribution. This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes.

This problem is predominant in scenarios where anomaly detection is crucial like electricity pilferage, fraudulent transactions in banks, identification of rare diseases, etc. In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.

This happens because Machine Learning Algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.

This guide describes various approaches for solving such class imbalance problems using various sampling techniques. We also weigh each technique for its pros and cons. Finally, I reveal an approach using which you can create a balanced class distribution and apply ensemble learning technique designed especially for this purpose.

 

Table of Content
Challenges faced with Imbalanced datasets
Approach to handling Imbalanced Datasets
Illustrative Example
Conclusion


1. Challenges faced with imbalanced data
One of the main challenges faced by the utility industry today is electricity theft. Electricity theft is the third largest form of theft worldwide. Utility companies are increasingly turning towards advanced analytics and machine learning algorithms to identify consumption patterns that indicate theft.

However, one of the biggest stumbling blocks is the humongous data and its distribution. Fraudulent transactions are significantly lower than normal healthy transactions i.e. accounting it to around 1-2 % of the total number of observations. The ask is to improve identification of the rare minority class as opposed to achieving higher overall accuracy.

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.

Example of imbalanced data
Let’s understand this with the help of an example.

Ex: In an utilities fraud detection data set you have the following data:

Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate= 2 %

The main question faced during data analysis is – How to get a balanced dataset by getting a decent number of samples for these anomalies given the rare occurrence for some them?

Challenges with standard Machine learning techniques
The conventional model evaluation methods do not accurately measure model performance when faced with imbalanced datasets.

Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

Evaluation of a classification algorithm performance is measured by the Confusion Matrix which contains information about the actual and the predicted class.


Accuracy of a model = (TP+TN) / (TP+FN+FP+TN)

However, while working in an imbalanced domain accuracy is not an appropriate measure to evaluate model performance. For eg: A classifier which achieves an accuracy of 98 % with an event rate of 2 % is not accurate, if it classifies all instances as the majority class. And eliminates the 2 % minority class observations as noise.

Examples of imbalanced data
Thus, to sum it up, while trying to resolve specific business challenges with imbalanced data sets, the classifiers produced by standard machine learning algorithms might not give accurate results. Apart from fraudulent transactions, other examples of a common business problem with imbalanced dataset are:

Datasets to identify customer churn where a vast majority of customers will continue using the service. Specifically, Telecommunication companies where Churn Rate is lower than 2 %.
Data sets to identify rare diseases in medical diagnostics etc.
Natural Disaster like Earthquakes
Dataset used
In this article, we will illustrate the various techniques to train a model to perform well against highly imbalanced datasets. And accurately predict rare events using the following fraud detection dataset:

Total Observations = 1000

Fraudulent   Observations =20

Non-Fraudulent Observations = 980

Event Rate= 2 %

Fraud Indicator = 0 for Non-Fraud Instances

Fraud Indicator = 1 for Fraud

 

2. Approach to handling Imbalanced Data
2.1 Data Level approach: Resampling Techniques
Dealing with imbalanced datasets entails strategies such as improving classification algorithms or balancing classes in the training data (data preprocessing) before providing the data as input to the machine learning algorithm. The later technique is preferred as it has wider application.

The main objective of balancing classes is to either increasing the frequency of the minority class or decreasing the frequency of the majority class. This is done in order to obtain approximately the same number of instances for both the classes. Let us look at a few resampling techniques:

 

### 2.1.1  Random Under-Sampling
Random Undersampling aims to balance class distribution by randomly eliminating majority class examples.  This is done until the majority and minority class instances are balanced out.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are taking 10 % samples without replacement from Non Fraud instances.  And combining them with Fraud instances.

Non Fraudulent Observations after random under sampling = 10 % of 980 =98

Total Observations after combining them with Fraudulent observations = 20+98=118

Event Rate for the new dataset after under sampling = 20/118 = 17%

 

- Advantages
It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.
- Disadvantages
It can discard potentially useful information which could be important for building rule classifiers.
The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.
 

### 2.1.2  Random Over-Sampling
Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

In this case we are replicating 20 fraud observations   20 times.

Non Fraudulent Observations =980

Fraudulent Observations after replicating the minority class observations= 400

Total Observations in the new data set after oversampling=1380

Event Rate for the new data set after under sampling= 400/1380 = 29 %

- Advantages
Unlike under sampling this method leads to no information loss.
Outperforms under sampling
- Disadvantages
It increases the likelihood of overfitting since it replicates the minority class events.
 

### 2.1.3  Cluster-Based Over Sampling
In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class have an equal number of instances and all classes have the same size.  

Total Observations = 1000

Fraudulent   Observations =20

Non Fraudulent Observations = 980

Event Rate= 2 %

Majority Class Clusters
Cluster 1: 150 Observations
Cluster 2: 120 Observations
Cluster 3: 230 observations
Cluster 4: 200 observations
Cluster 5: 150 observations
Cluster 6: 130 observations
Minority  Class Clusters
Cluster 1: 8 Observations
Cluster 2: 12 Observations
 

After oversampling of each cluster, all clusters of the same class contain the same number of observations.

Majority Class Clusters
Cluster 1: 170 Observations
Cluster 2: 170 Observations
Cluster 3: 170 observations
Cluster 4: 170   observations
Cluster 5: 170   observations
Cluster 6: 170   observations
Minority   Class Clusters
Cluster 1: 250 Observations
Cluster 2: 250 Observations
Event Rate post cluster based oversampling sampling = 500/ (1020+500) = 33 %

Advantages
This clustering technique helps overcome the challenge between class imbalance. Where the number of examples representing positive class differs from the number of examples representing a negative class.
Also, overcome challenges within class imbalance, where a class is composed of different sub clusters. And each sub cluster does not contain the same number of examples.
Disadvantages
The main drawback of this algorithm, like most oversampling techniques is the possibility of over-fitting the training data.
 

### 2.1.4  Informed Over Sampling: Synthetic Minority Over-sampling Technique for imbalanced data
This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.

Total Observations = 1000

Fraudulent  Observations = 20

Non Fraudulent Observations = 980

Event Rate = 2 %

A sample of 15 instances is taken from the minority class and similar synthetic instances are generated 20 times

Post generation of synthetic instances, the following data set is created

Minority Class (Fraudulent Observations) = 300

Majority Class (Non-Fraudulent Observations) = 980

Event rate= 300/1280 = 23.4 %

 

-Advantages
Mitigates the problem of overfitting caused by random oversampling as synthetic examples are generated rather than replication of instances
No loss of useful information
- Disadvantages
While generating synthetic examples SMOTE does not take into consideration neighboring examples from other classes. This can result in increase in overlapping of classes and can introduce additional noise
SMOTE is not very effective for high dimensional data
Imbalanced classification technique

  

###2.1.5  Modified synthetic minority oversampling technique (MSMOTE) for imbalanced data
It is a modified version of SMOTE. SMOTE does not consider the underlying distribution of the minority class and latent noises in the dataset. To improve the performance of SMOTE a modified method MSMOTE is used.

This algorithm classifies the samples of minority classes into 3 distinct groups – Security/Safe samples, Border samples, and latent nose samples. This is done by calculating the distances among samples of the minority class and samples of the training data.

Security samples are those data points which can improve the performance of a classifier. While on the other hand, noise are the data points which can reduce the performance of the classifier.  The ones which are difficult to categorize into any of the two are classified as border samples.

While the basic flow of MSOMTE is the same as that of SMOTE (discussed in the previous section).  In MSMOTE the strategy of selecting nearest neighbors is different from SMOTE. The algorithm randomly selects a data point from the k nearest neighbors for the security sample, selects the nearest neighbor from the border samples and does nothing for latent noise.
