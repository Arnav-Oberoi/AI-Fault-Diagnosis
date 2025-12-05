% -------------------------------------------------------------------
% ---  Diagnostic Script for Low Accuracy Machine Learning Model  ---
% -------------------------------------------------------------------
% This script will help you diagnose why your model's accuracy is low by
% checking for class imbalance, weak features, and poor model tuning.
%
% Instructions:
% 1. Place your CSV file in the same folder as this script.
% 2. Change 'your_data.csv' in Step 1 to your actual file name.
% 3. Run the script and analyze the output in the Command Window and
%    the generated plots.
% -------------------------------------------------------------------

clear; clc; close all;

%% Step 1: Load and Prepare Data
disp('--- Step 1: Loading Data ---');
try
    fileName = 'C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\Data\TrainingData\merged_data.csv'; % <-- CHANGE THIS TO YOUR FILENAME
    data = readtable(fileName);
    
    X = data(:, 1:3);
    Y_table = data(:, 4);
    Y = categorical(Y_table{:,1});
    
    fprintf('Data ''%s'' loaded successfully.\n\n', fileName);
catch ME
    error('Failed to load or process the CSV file. Please check the file name and format.');
end

%% Step 2: Diagnostic #1 - Check for Class Imbalance
disp('--- Step 2: Diagnostic #1 - Checking for Class Imbalance ---');
figure;
histogram(Y);
title('DIAGNOSTIC 1: Class Distribution');
ylabel('Number of Samples per Class');

% Display counts in command window
disp('Sample counts per class:');
countcats(Y)

fprintf('\n>>> INTERPRETATION (Class Imbalance):\n');
fprintf('Look at the histogram plot and the counts above.\n');
fprintf('IF: Bars are very different heights (e.g., one class has thousands of samples, others have hundreds).\n');
fprintf('THEN: You have a highly imbalanced dataset. This is a major problem and likely a primary cause of low accuracy.\n');
fprintf('-------------------------------------------------------------\n\n');
pause(2); % Pause to allow user to read

%% Step 3: Diagnostic #2 - Check Feature Importance
disp('--- Step 3: Diagnostic #2 - Checking Feature Importance ---');
fprintf('Training a basic Random Forest to evaluate feature predictive power...\n');

% Train a simple model
Mdl_for_importance = fitcensemble(X, Y, 'Method', 'Bag', 'Learners', 'Tree');

% Calculate feature importance
imp = oobPermutedPredictorImportance(Mdl_for_importance);

% Plot the results
figure;
bar(imp);
title('DIAGNOSTIC 2: Original Feature Importance');
xlabel('Feature Number');
ylabel('Out-of-Bag Permuted Importance');
xticks(1:width(X));
xticklabels(X.Properties.VariableNames);

fprintf('\n>>> INTERPRETATION (Feature Importance):\n');
fprintf('This plot shows how much accuracy drops if a feature is removed. Higher bars are better.\n');
fprintf('IF: All bars are very low (close to zero).\n');
fprintf('THEN: Your original features are weak and do not have much predictive power on their own. You should focus on "Feature Engineering".\n');
fprintf('IF: One bar is much higher than others.\n');
fprintf('THEN: Only one of your features is useful to the model.\n');
fprintf('-------------------------------------------------------------\n\n');
pause(2); % Pause to allow user to read

%% Step 4: Diagnostic #3 - Find Best Possible Accuracy with Tuning
disp('--- Step 4: Diagnostic #3 - Finding Best Possible Accuracy ---');
fprintf('This will take a few minutes. A plot will appear showing the optimization process.\n');
fprintf('The goal is to see the highest accuracy the model can achieve with the current features.\n\n');

% Partition data for a fair test
rng(1);
cv = cvpartition(Y, 'HoldOut', 0.20);
idx = cv.test;
XTrain = X(~idx, :);
YTrain = Y(~idx, :);
XTest  = X(idx, :);
YTest  = Y(idx, :);

% Use automatic hyperparameter optimization to find the best model
OptimizedMdl = fitcensemble(XTrain, YTrain, ...
                   'Method', 'Bag', ...
                   'Learners', 'Tree', ...
                   'OptimizeHyperparameters', 'auto', ...
                   'HyperparameterOptimizationOptions', struct(...
                        'ShowPlots', true, ...
                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                        'Verbose', 0)); % Set Verbose to 0 to keep command window clean

% Evaluate the BEST model found during optimization
loss = loss(OptimizedMdl, XTest, YTest);
best_accuracy = 100 * (1 - loss);

fprintf('\n>>> INTERPRETATION (Best Possible Accuracy):\n');
fprintf('The automated tuning process has completed.\n');
fprintf('The best accuracy found for your dataset with these features is: %.2f%%\n', best_accuracy);
fprintf('IF: This "best accuracy" is still low (e.g., 45-55%%).\n');
fprintf('THEN: This is a strong sign that the problem is NOT the model''s settings, but the DATA itself (weak features or imbalance).\n');
fprintf('IF: The accuracy is now high (e.g., >80%%).\n');
fprintf('THEN: The problem was simply that the model needed tuning.\n');
fprintf('-------------------------------------------------------------\n\n');

disp('DIAGNOSIS COMPLETE.');