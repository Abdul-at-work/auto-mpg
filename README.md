# auto-mpg

Task 1. Implementing the k-NN algorithm.
In this part of the assignment, you will implement the well-known k-NN algorithm yourself in Python.
Remember that it is an instance based learning algorithm including the following main steps:
• Given the instance whose label to be predicted (xq), find k nearest neighbors of xq within the training
data set. At this step, you are not expected to implement an index structure, just calculate the
similarities and find k nearest neighbors.
• Decide for the label of xq according to the majority vote for predicting categorical labels and weighted
mean for numerical labels.


Task 2. Applying the k-NN algorithm.
In this part of the assignment, you will apply your algorithm to a real-world dataset.
• You will use the ’Auto MPG’ dataset. Dataset is provided for you as an attachment in ODTUCLASS. You can find more information about the dataset in the following link. However, please
download the attached dataset as it is slightly different from the one in the link. https://archivebeta.ics.uci.edu/ml/datasets/auto+mpg
• You will predict the ’mpg’ (fuel consumption in miles per gallon) of cars.
– Report the result for k values from 2 to 10, under 3-fold cross validation. Since your prediction
results are numeric values, report the prediction performance in terms of MSE, RMSE and MAPE.
– Report the prediction time for all cases.


Task 3. Comparing with the classifiers in the scikit-learn library.
In this part of the assignment, compare the performance of your best implementation with the following
supervised learning methods in scikit-learn library: KNeighborsRegressor and DecisionTreeRegressor.Note
that you may need to adapt the domains of the attributes according to the classifier. You do not need to
optimize the parameters for these classifiers. Just use the default settings.
– Report the results under 3-fold cross validation in terms of MSE, RMSE and MAPE.
– Report the prediction time for all cases.

