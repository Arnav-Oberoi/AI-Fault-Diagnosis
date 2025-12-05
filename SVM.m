%% Multi-class SVM from CSV Dataset
% Handles 3-class labels: normal, wear, backlash
% Includes normalization and hyperparameter optimization

clc; clear; close all;

%% Step 1: Load Dataset
data = readtable('C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\Data\TrainingData\merged_data.csv');

% Assuming columns 1-3 are numeric features, column 4 is label
X = data{:, 1:3};
Y = categorical(data{:, 4});

%% Step 2: Split Data (80% Train, 20% Validation)
cv = cvpartition(Y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest  = X(test(cv), :);
YTest  = Y(test(cv), :);

%% Step 3: Normalize Features
mu = mean(XTrain);
sigma = std(XTrain);
XTrain = (XTrain - mu) ./ sigma;
XTest  = (XTest - mu) ./ sigma;

%% Step 4: Train Multi-class SVM (ECOC) with Hyperparameter Optimization
disp('Training multi-class SVM with hyperparameter optimization...');

cvInner = cvpartition(YTrain, 'KFold', 5); % for internal CV during optimization

t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);

svmModel = fitcecoc(XTrain, YTrain, ...
    'Learners', t, ...
    'Coding', 'onevsone', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct( ...
        'CVPartition', cvInner, ...      % internal 5-fold CV
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', false, ...
        'Verbose', 0, ...
        'UseParallel', false, ...        % disables parallel toolbox dependency
        'MaxObjectiveEvaluations', 30));

%% Step 5: Evaluate on Test Set (20%)
YPred = predict(svmModel, XTest);
testAccuracy = mean(YPred == YTest) * 100;
fprintf('\nValidation Accuracy (20%% hold-out): %.2f%%\n', testAccuracy);

%% Step 6: Confusion Matrix
figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Multi-class SVM Results';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% Step 7: Display Best Hyperparameters
disp('Best hyperparameters found:');
disp(svmModel.HyperparameterOptimizationResults.XAtMinObjective);
