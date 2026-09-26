# Heart-Disease-Prediction-Using-Machine-Learning
# Problem Statement
Heart disease is a leading cause of death worldwide, which is why it is essential that it be detected accurately in its early stages to have effective treatment and prevention. The traditional diagnosis has relied upon medical expertise, clinical trials, and patient history, all of which is often time consuming and susceptible to human error.The goal of this particular project is to develop a prediction system based on machine learning to diagnose whether the given patient will get heart disease or not, based on clinical and physiological characteristics. The data set contains features like age, sex, type of chest pain (cp), resting blood pressure (trestbps), total cholesterol level (chol), Fasting Blood Sugar (fbs), maximum heart rate during an exercise test (thalach), exercise-induced angina (exang), depression in the ST segment (oldpeak) and the percentage of peak exercise ST segment slope (slope). Other features were also include that help differentiate whether the given patient will get heart disease or not such as the number of major vessels colored by fluoroscopy (ca) and Type of Thalassemia (thal).In this system we will use these input variables to predict whether the given patient will get heart disease or not based on these given characteristics. We will use machine learning models to discover the patterns in data and their relation so that we will be able to help doctor to predict the disease in very less time and more precisely.
The main issues in this system are dealing with the missing and skewed data, feature selection and achieving the appropriate level of prediction accuracy.
#  Research Objectives and Methodology 

# Research Objectives
1)To identify and preprocess relevant clinical and demographic features

# Objective:
Select key risk factors (e.g., age, blood pressure, cholesterol, smoking) from available datasets that are strongly correlated with heart disease.

2)To evaluate various machine learning algorithms for predictive accuracy

# Objective: 
Compare the performance of models such as Logistic Regression, Decision Trees, Random Forests, SVM, XG Boost, and Neural Networks.

3) To develop a robust prediction model for early detection of heart disease

# Objective: 
Create a machine learning model with high accuracy, precision, and recall, suitable for real-world implementation in healthcare settings.

# Research Problem
Heart  disease is still among the major causes of death globally, and it has a huge cost to healthcare systems and society as a whole. Early and correct diagnosis is crucial for successful treatment and prevention, but conventional methods of diagnosis tend to be slow, expensive, and may be lacking in predictive value, particularly in the initial stages of the disease. With expanding access to healthcare data, machine learning (ML) methodologies hold the promise to enhance the forecasting of heart disease by revealing subtle patterns and associations within clinical information that are perhaps not readily apparent to human experts. Yet, obstacles exist in the decision of which features are most salient to use, selection of suitable ML algorithms, generation of interpretable models, and high prediction accuracy across heterogeneous populations.

# Data Collection
The dataset which we have used is available on Kaggle  website  and  is  available for public download. It is processed from UCI’s dataset and contains valid tuples for further processing.
                              Dataset Characters 	Multivariate 
                              Number of Tuples 	1025
                              Number of Attributes	14
                             Attribute Datatype 	Categorical Integer, Real
                                      Source 	Kaggle 
                                       Dataset Detail
Next after downloading datasets is generally to clean the data if some missing values, noise in data is present. First the data is imported from downloaded csv files using pandas libraries of Python, to RAM in data type called data frame which is optimized adequately to handle two dimensional array data. The dataset does not have any null values. 14 of our attributes are the attributes which are used to predict the result, while the last attribute “target” is the result, i.e., whether or not the individual was suffering from heart  disease.
The following are the result of the heartData.info() and heartData.describe() command, which reflects the statistics of Processed dataset.

 <img width="703" height="279" alt="image" src="https://github.com/user-attachments/assets/f4115574-6090-4fc0-8d25-699df7784914" />
                                        Attributes in Heart Dataset with datatype

  <img width="613" height="265" alt="image" src="https://github.com/user-attachments/assets/c258d7f5-a856-404e-a4b7-f09ca91184dd" />
                                         Detailed Description of Heart Disease Dataset -1

  <img width="610" height="320" alt="image" src="https://github.com/user-attachments/assets/b3f38676-1a94-48f3-8e9d-c8a9ead85c0a" />
                                      Detailed Description of Heart Disease Dataset -2
                                        
# Checking Data Distribution
Following is the categorization of dataset based on target  class:
<img width="490" height="316" alt="image" src="https://github.com/user-attachments/assets/a159ca45-1af5-4be2-bd42-6eb35f54c24b" />
                          Categorization of dataset based on target class

Study of Dataset : We will generate and understand correlation between our attributes and target class. Correlation matrix will be generated and plotted using  matplotlib. For a correlation matrix, the more positive the value of correlation, the more increase in value of one variable causes the other to increase i.e. more directly proportional. The higher a negatively correlated variable gets, the lower the value of the target  becomes.

<img width="524" height="248" alt="image" src="https://github.com/user-attachments/assets/a134d754-dae5-443f-9906-65c53a83650e" />
     Maximum positive correlated features is cp and thalach and maximum negative correlated features is exang and old peak 
                                   
                                   Correlation Diagram as heatmap
Using the correlation matrix, we discover the target to be positively correlated to chest pain significantly. This arrival should also be obvious as the greater amount of chest pain means greater risk in the heart. Max Heart rate is also significantly correlated to Target for the reason that healthier hearts do not need to become elevated  much  for  blood supply. It simply means higher heart rate, higher the risk of heart disease. A positive correlation can be observed between those with thalassemia. Since it is  a  3  valued ordinal, where 3 indicates normal, 6 to 7 defects. Hence being in the normal category is better. Presence of negative correlation among target and angina can also be observed. This observation also agrees with common sense as exercise causes muscles to crave for more oxygen, in-turn boosting heartbeat, while narrowed-down arteries would act as blockage.
       
  <img width="411" height="327" alt="image" src="https://github.com/user-attachments/assets/80b25f57-6931-4792-8a80-02cd28f01336" />

# User Interactive Front End
There will be a multipage website containing a homepage, a page for the user in which he enters details to predict the presence or absence of heart disease and a page about us. Flask has been used to connect the frontend with trained models of the  backend. Flask is a Python web framework that was created with a philosophy in mind. Armin Ronacher   conceived and developed Flask as an April Fool's Day hoax in 2010. Despite its comedic beginnings, the Flask framework has grown in popularity as a viable alternative to   Django projects' monolithic structure and dependencies.There are some advantages of flask over other frameworks as per requirement of our project. They are,Flask is a Python web framework built for rapid development of small projects, Flask offers a diversified working style while Django offers a Monolithic working  style.

# System Approach
HARDWARE REQUIREMENT
The hardware requirements for running this website and model  are:
●	RAM – 8.00 GB
●	Operating System – Windows 11 
●	Processor – Intel(R) Core(TM) i3-1115g4
●	Processor speed – 3.00 GHz
SOFTWARE REQUIREMENTS
The programming language used to develop this application is Python and the IDE used is Jupyter Notebook. Front end is made using HTML, CSS and is integrated with  flask.
●	Programming Language – Python
●	Python IDE – Jupyter Notebook
●	Python Libraries: Flask

 #  Data analysis, results, and interpretation
The models used are the following:
●	Logistic Regression
●	K Nearest Neighbors
●	Support Vector Machine
●	Decision Tree
●	Random Forest

# Logistic Regression
This is the one of the most common model used in ML ,Logistic Regression is often applied in the actual manufacturing context the fields such as data mining, automatic disease diagnosis and economic prediction.
For our model,   use Logistic regression to know the risk factors for heart disease and forecast the probability of disease occurrence based on risk factors. This model is most frequently applied for classification, primarily two-category issues (that is, there are only two types of output, each representing one category), and can indicate the probability of occurrence of each classification event. Logistic regression model is shown below:
This technique used is also known as sigmoid function .Sigmoid function helps in the easy representation in graphs. Logistic regression also provides better accuracy. By using equation the logistic regression algorithm is represented in the graphs showing the difference between the attributes.
 Where Y refers to binary dependent variable (Y is equal to 1 if event happens; Y=0 otherwise), e stands for the foundation of natural logarithms and Z  means
with constant β0 ,coefficents β j and predictors X j , for p  predictors(j=1,2,3,.....,p)
with constant β0 ,coefficients β j and predictors X j , for p  predictors(j=1,2,3,.....,p)
The process of modeling the probability of a discrete outcome given an input variable is known as the Logistic Regression. The most common logistic regression, as its name suggests is not regression rather it is a classification algorithm that classifies something that can take two values such as true/false, yes/no, and so on. Logistic regression identifies a hyperplane in a manner that when it is passes through a function whose value ranges between 0 and 1 (typically we use sigmoidal), it optimizes cost function. Bases on closeness to 0 or 1, it predicts a Boolean output. Here vector parameters is used for training. σ(.) is usually a sigmoid function, with output between 0 and 1.

# Features of Logistic Regression:
Multinomial logistic regression is the type of regression which uses the softmax function to compute probabilities.
● We use loss function to learn weights(vector w and bias b) from a labeled training. we perform such activity to minimize the cross-entropy loss.
● Iterative algos like gradient descent are used to get the weight(optimal).while minimizing the loss function the type of convex optimization problem.
● To avoid overfitting regularization is used.
● Logistic regression has the ability to transparently study the importance of individual features.

# Advantages and Disadvantages of Logistic Regression
Advantages
● This technique is perform well and fast where we have to classify unknown records.
● This is not limited to binary classification we can easily extend it to  multinomial regression.
Disadvantages
● Logistic regression will not perform well If the number of observations is lesser than the number of features in such condition it may lead to overfitting.
● Logistic regression constructs the linear boundaries.
In the logistic function equation, x is the input variable. Let's feed in values −20 to 20 into the logistic function. As illustrated in Figure the inputs have been transferred to between 0 and  1.

 <img width="594" height="322" alt="image" src="https://github.com/user-attachments/assets/e45d8ac9-a098-4588-bd50-d87ad3b8132d" />
                                        Sigmoid Graph                  

<img width="841" height="210" alt="image" src="https://github.com/user-attachments/assets/6afd11c6-4c70-4cb1-a429-06beab1c4311" />
                                       Logistic Model Result 
                                             Accuracy 80%
# KNN
K-Nearest Neighbors (KNN) is a classification and regression online (lazy) learning algorithm. It's a non-parametric approach to the extent that it doesn't make any assumption regarding the distribution of the data.
# How It Works:
Input Data: To predict a given new data point, KNN computes the similarity between this data point and the entire training data set.
Distance Metric: Any distance metric such as Euclidean, Manhattan, or others can be employed based on the problem.
Finding Neighbors: It finds the k training samples nearest to the input point.
Prediction:
For classification, the most common label among the k neighbors is returned.
For regression, the mean value of the k neighbors is taken as the prediction.
Model Training & Prediction (Using sklearn):
Training: Employ the fit() function of K Neighbors Classifier or KNeighborsRegressor from sklearn .neighbors.
Prediction: Employ the predict() function on the test set.
Selection of k value:
Selecting k (number of neighbors) is important:
A low k may produce overfitting (high variance).
A high k can smoothen the decision boundary but may lead to underfitting (high bias).
For selection of the best k, vary k and check the model performance using cross-validation.
The location at which adding k no longer further enhances accuracy appreciably is referred to as the knee point.
# Features of KNN
•	Supervised Learning: KNN is a supervised machine learning algorithm, meaning it relies on a labeled dataset to learn and make predictions.
•	Simplicity :It is one of the simplest and most intuitive machine learning algorithms, making it easy to implement and understand.
•	Instance-Based Learning: KNN is a lazy learner; it doesn’t learn a discriminative function from the training data but memorizes the training dataset instead.
•	Based on Feature Similarity: The core idea of KNN is that similar data points exist in close proximity. It assumes that the data points that are near each other are more likely to share the same output label.
•	Distance Metric: KNN classifies new data points based on a distance metric (e.g., Euclidean, Manhattan, Minkowski) by comparing it to its 'K' nearest neighbors in the training set.
•	Non-Parametric: KNN does not make any assumptions about the underlying data distribution, making it a flexible model for various data types.
•	Adaptability: It can be used for both classification and regression problems

  <img width="353" height="226" alt="image" src="https://github.com/user-attachments/assets/c54e60c5-52a1-4143-ba13-57edd9ba36ba" />
                                        KNN output graph

Since we are getting the best value at k = 11 and k = 15, it was decided in favour of k = 11, which is usually the default value. Finally, prediction is done on test data and the result table is prepared
                                        
<img width="496" height="228" alt="image" src="https://github.com/user-attachments/assets/b21c71b8-7471-4b24-a38e-0688baa48869" />
                                            Classification report of KNN
                                                  Accuracy 64%
# SVM
It belongs to the supervised machine learning algorithm. This is the type of algorithm that can be used for both classification and regression challenges. SVM is mostly used in classification problems. In the SVM algorithm, we set each data object as a point in the n- dimensional space (where n is the number of attributes you have) by the value of each element which is the value of a particular combination. Then, we do the splitting by finding a hyper-plane that separates the two sections very well. Support Vectors are simply links to individual points.
# SVM features:
●	Use missing as level
●	Include iterations report
●	Penalty in the svm we can specify the penalty value. The default value we use is 1.
●	Kernel —we can use various type of kernel in SVM.
●	Polynomial Degree — The default value is 2.
●	Tolerance — specifies the minimum number at which the iteration stops. The default value is 0.000001.
●	Maximum iterations — In svm this indicate maximum number of iterations that is allowed with each try. The default value is 25.

  <img width="401" height="228" alt="image" src="https://github.com/user-attachments/assets/2e6cbeab-959c-4f17-9a49-35c98a4c263a" />
  <img width="463" height="230" alt="image" src="https://github.com/user-attachments/assets/e040d73e-a490-4c34-a573-2b13c2c7be1b" />
                                  SVM Scores against various kernels
                                  Accuracy: 79% with linear Kernel

                                 Kernel	            Accuracy
                                 Linear	               79
                                  Poly	               69
                                   RBF	               66
                                 Sigmoid	             55

                               SVM score against Various  Kernel

 # Decision Tree
A decision tree contains a flowchart-like structure. In the structure of DT(Decision Tree) in a test, an attribute is represented using an internal node. The result of the test is represented using the branch. To represent a class label, a leaf node is used. To represent classification rules paths from root to leaf are used. This is the type of analysis in which closely related influence diagrams are used for a visual and analytical root. Where the expected values of competing alternatives are calculated.

# Architecture of Decision Tree.
    
  <img width="714" height="402" alt="image" src="https://github.com/user-attachments/assets/57c72fd8-9180-4fed-a73b-a25cf772e549" />
                                          Decision Tree image    

   <img width="431" height="271" alt="image" src="https://github.com/user-attachments/assets/7f3cb4c5-6b67-44ab-b6f2-f18816cdfb5d" />
                                             Decision tree Result Graph

# Random Forest Classifier
A Random Forest Classifier is a method of ensemble learning applied to classification problems. It constructs multiple decision trees and combines their predictions to enhance accuracy and avoid overfitting. Each tree is trained on a random subset of data and attributes, hence making the model stable, robust, and effective in handling complex datasets.

 <img width="454" height="295" alt="image" src="https://github.com/user-attachments/assets/3c1cd44c-df9c-47cc-b079-fc0d46e59cfc" />
                                    Random Forest Image

# Random Forest Features
●	It runs Efficiently in scenarios where database is very huge.
●	We can perform classification on Thousands of Input variables.
●	Using this we can find which variable is useful for our classification.
●	Using this we can easily calculate the missing data and also maintain the good accuracy. Even though it has missing data.

  <img width="765" height="433" alt="image" src="https://github.com/user-attachments/assets/3adc1fcd-b690-4732-82d7-04a2a827b801" />
  <img width="762" height="380" alt="image" src="https://github.com/user-attachments/assets/cd6502b8-8282-40b2-a8f8-f9f0902727bc" />
                                       Random Forest Result Image
Accuracy: 82% with 100 estimators. Increase in number of  estimators  will  increase the calculation complexities significantly

# Findings and Conclusion

# Findings 

# Working Model Diagram
The working front End has a page which contains form where user will enter all the required medical information
The form page screenshot is attached as:

<img width="759" height="357" alt="image" src="https://github.com/user-attachments/assets/cb44533c-bf5c-4ca5-b866-8b0baa5fe2fc" />
<img width="753" height="431" alt="image" src="https://github.com/user-attachments/assets/381383b3-941b-4c62-b4e5-5cbd0dceb018" />
                                           Input Form

# The result page screenshot:
   <img width="793" height="381" alt="image" src="https://github.com/user-attachments/assets/5d8c68e4-10e7-4318-9836-db80e4c8adfa" />
                                            Result Page 
It also has a button to print the report generated so that user can have a record of data entered by him/her and the respective result.

# Conclusion
Anyone can verify the existence or non-existence of heart disease from their devices with only assistance of medical records. Medical professionals can use this tool to directly determine the results. Decrease in engagement of medical professionals for decision making. The project can be implemented on a large scale and be modified in all hospitals about heart diagnosis centers.

# Future Scope
1. Integration with Wearable Technology 
Machine learning algorithms can utilize continuous streams of data from wearable devices (e.g., fitness trackers, smartwatches) to offer real-time heart health information and risk prognostications.
2. Personalized Medicine
Patient-specific data like genetics, medical history, and lifestyle factors can be analyzed by ML algorithms to develop personalized preventive measures or treatment plans for heart disease.
3. Multimodal Data Utilization
Merging clinical information (e.g., lab results, medical histories) with imaging information (e.g., echocardiograms, CT scans) might result in more precise and complete prediction models.
4. Improved Interpretability
Next-generation models can prioritize explainability by using methodologies such as SHAP (Shapley Additive Explanations) or LIME (Local Interpretable Model-Agnostic Explanations) so that doctors and patients can comprehend predictions and have confidence in AI tools.
5. AI-Driven Research
Machine learning can reveal latent patterns in large datasets, pushing research into new biomarkers or uninvestigated causes of heart disease.
6. Global Accessibility
Create models optimized for low-resource settings, making cost-effective and accessible heart disease prediction tools available to disadvantaged groups worldwide.
7. Dynamic Monitoring Systems
Implement systems for ongoing health monitoring, where predictive models evolve over time as new information becomes available, providing current risk assessments.
8. Integration into Telemedicine
Machine learning-based prediction functionalities can be embedded within telemedicine platforms, supporting remote consultations and healthcare extension.
9. Behavioral Prediction and Modification
New systems might not only predict heart disease but also evaluate and promote healthier behaviors to actively lower risk factors.
10. Advanced Model Architectures
The use of deep learning methodologies, such as transformer models or graph neural networks (GNNs), can also improve predictive accuracy and resilience.

# Limitations
This project models require 13 attributes for their prediction. If we analyze the attributes closely, then most of the attributes are not available to any normal person until he/she takes some medical tests which will cost them more money, time, and medical professionals, and equipment. The attributes required are also more in medical terms than in general language which almost everyone can understand. It would be more friendly and easy for a user if the attributes that require more medical tests could be decreased significantly and more common attributes that are responsible for heart disease could be included. Some attributes that can work for this are whether the user is a smoker, alcoholic, exercise frequency, etc.
