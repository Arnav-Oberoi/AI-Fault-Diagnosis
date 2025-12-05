% --- MATLAB Script for Training an OPTIMIZED SVM Model ---
%
% This script loads a dataset, scales the features, and then uses
% Bayesian optimization to automatically find the best hyperparameters
% for a multi-class Support Vector Machine (SVM) to achieve the highest
% possible accuracy.
%
% =========================================================================
% --- 1. Load and Prepare the Data ---
% =========================================================================

% --- Specify the file path ---
filePath = 'C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\Data\TrainingData\merged_data.csv';

% --- Read the data from the CSV file ---
disp('Loading data...');
try
    dataTable = readtable(filePath);
catch ME
    error('Failed to read the file. Please check the file path. Original error: %s', ME.message);
end

% --- Display a preview of the data ---
disp('Data loaded successfully.');
head(dataTable)

% =========================================================================
% --- 2. Separate Predictors (Features) and Response (Labels) ---
% =========================================================================

% --- Isolate the feature and label columns ---
X = table2array(dataTable(:, 2:3)); % Predictor variables (features)
Y = table2array(dataTable(:, 4));   % Response variable (labels)

disp('Data has been separated into predictors (X) and response (Y).');

% =========================================================================
% --- 3. Split Data into Training and Testing Sets ---
% =========================================================================

% --- Create a random partition ---
disp('Splitting data into training (80%) and testing (20%) sets...');
cv = cvpartition(size(X, 1), 'HoldOut', 0.20);
idx = cv.test;

% --- Create the final training and test sets ---
XTrain = X(~idx,:);
YTrain = Y(~idx,:);
XTest  = X(idx,:);
YTest  = Y(idx,:);

fprintf('Training set size: %d rows\n', size(XTrain, 1));
fprintf('Testing set size: %d rows\n', size(XTest, 1));

% =========================================================================
% --- 3b. Scale the Feature Data ---
% =========================================================================
% CRITICAL STEP FOR SVM PERFORMANCE
disp('Scaling training and testing data...');

% --- Calculate mean and standard deviation from the TRAINING set ONLY ---
mu = mean(XTrain);
sigma = std(XTrain);

% --- Apply the scaling to both the training and testing sets ---
XTrain = (XTrain - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

disp('Data scaling complete.');

% =========================================================================
% --- 4. Find Best Hyperparameters via Optimization ---
% =========================================================================
% To get high accuracy, we automatically search for the best model settings.
disp('Starting hyperparameter optimization...');

% --- Define the SVM learner template ---
t = templateSVM('KernelFunction', 'rbf');

% --- Define the hyperparameters to optimize ---
% 'BoxConstraint' and 'KernelScale' are the key parameters for an RBF SVM.
% We give them a wide range to search within.
vars = [optimizableVariable('BoxConstraint', [1e-3, 1e3], 'Transform', 'log');
        optimizableVariable('KernelScale', [1e-3, 1e3], 'Transform', 'log')];

% --- Run Bayesian Optimization ---
% 'fitcecoc' will now repeatedly train on the data, automatically tuning
% the hyperparameters defined in 'vars' to minimize classification error.
% The returned object 'Mdl' is the final model trained with the best parameters.
Mdl = fitcecoc(XTrain, YTrain, 'Learners', t, ...
    'OptimizeHyperparameters', vars, ...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 5, ... % Limit iterations to 5 for a quick test
        'ShowPlots', true, ...
        'Verbose', 1));

disp('Optimization complete.');

% =========================================================================
% --- 5. Evaluate the Final, Optimized Model ---
% =========================================================================

disp('Evaluating the final optimized model on the test set...');

% --- Make predictions ---
% Use the final model 'Mdl' returned from the optimization process.
YPred = predict(Mdl, XTest);

% --- Calculate accuracy ---
accuracy = sum(strcmp(YPred, YTest)) / numel(YTest);

% --- Display the final accuracy ---
fprintf('\nFinal Optimized Model Accuracy: %.2f%%\n', accuracy * 100);

% =========================================================================
% --- 6. Visualize Performance with a Confusion Matrix ---
% =========================================================================

disp('Displaying the confusion matrix for the optimized model...');

figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix for Optimized SVM';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

