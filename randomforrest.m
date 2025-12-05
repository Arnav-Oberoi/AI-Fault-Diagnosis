% --- MATLAB Script for Training an OPTIMIZED Boosted Trees Model ---
%
% This script uses an AdaBoost ensemble with Bayesian optimization to
% automatically tune hyperparameters for the highest possible accuracy.
%
% =========================================================================
% --- 1. Load and Prepare the Data (from .mat file) ---
% =========================================================================

% --- Specify the file path (CHANGED to .mat) ---
filePath = 'C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\Scripts\actuator_training_data.mat';

% --- Load the data from the .mat file ---
disp('Loading data from .mat file...');
try
    % The load function creates a struct containing all variables from the file.
    dataStruct = load(filePath);

    % --- Automatically get the variable name from the loaded struct ---
    % This assumes your .mat file contains only one primary data variable.
    variableNames = fieldnames(dataStruct);
    dataTable = dataStruct.(variableNames{1});

    % --- Ensure data is a table for consistency ---
    % If your .mat file stores a simple matrix, this converts it to a table.
    if ~istable(dataTable)
        disp('Data was a matrix. Converting to table format.');
        dataTable = array2table(dataTable);
    end

catch ME
    error('Failed to load the .mat file. Check the path and ensure it contains a valid data variable. Original error: %s', ME.message);
end


% --- Display a preview of the data ---
disp('Data loaded successfully.');
head(dataTable)

% =========================================================================
% --- 2. Separate Predictors (Features) and Response (Labels) ---
% =========================================================================

% --- Isolate the feature and label columns based on the new structure ---
X_original = table2array(dataTable(:, 2:5)); % Features are now columns 2, 3, 4, 5
Y = table2array(dataTable(:, 1));             % Labels are now in column 1

disp('Data has been separated into predictors (X) and response (Y).');

% =========================================================================
% --- 2b. ADVANCED Feature Engineering (UPDATED FOR 4 FEATURES) ---
% =========================================================================
% To improve accuracy, we create a more extensive set of new features.
disp('Performing ADVANCED feature engineering for 4 input features...');

X1 = X_original(:, 1); % Torque_RMS
X2 = X_original(:, 2); % Torque_Mean
X3 = X_original(:, 3); % Vibration_Std
X4 = X_original(:, 4); % Vibration_Max

% --- Create a new, rich set of features ---
% Pairwise interactions between all features
feat_inter_12 = X1 .* X2;
feat_inter_13 = X1 .* X3;
feat_inter_14 = X1 .* X4;
feat_inter_23 = X2 .* X3;
feat_inter_24 = X2 .* X4;
feat_inter_34 = X3 .* X4;

% Polynomial features (squares)
feat_sq_1 = X1.^2;
feat_sq_2 = X2.^2;
feat_sq_3 = X3.^2;
feat_sq_4 = X4.^2;

% Meaningful Ratios (add small constant to avoid division by zero)
feat_ratio_torque = X2 ./ (X1 + 1e-6); % Mean vs RMS Torque
feat_ratio_vib = X4 ./ (X3 + 1e-6);    % Max vs Std Vibration

% --- Combine original features with all new engineered features ---
X = [X_original, ...
     feat_inter_12, feat_inter_13, feat_inter_14, ...
     feat_inter_23, feat_inter_24, feat_inter_34, ...
     feat_sq_1, feat_sq_2, feat_sq_3, feat_sq_4, ...
     feat_ratio_torque, feat_ratio_vib];

fprintf('New, richer feature set created with %d total features.\n', size(X, 2));


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
% --- 4. Find Best Hyperparameters via Optimization (Using BOOSTING) ---
% =========================================================================
% To get high accuracy, we automatically search for the best model settings.
disp('Starting hyperparameter optimization for Boosted Trees (AdaBoost)...');

% --- Define the hyperparameters to optimize for a boosting model ---
vars = [optimizableVariable('MinLeafSize', [1, 20], 'Type', 'integer');
        optimizableVariable('NumLearningCycles', [50, 800], 'Type', 'integer');
        optimizableVariable('LearnRate', [0.0001, 1], 'Type', 'real', 'Transform', 'log')];

% --- Run Bayesian Optimization with AdaBoostM2 method ---
Mdl = fitcensemble(XTrain, YTrain, 'Method', 'AdaBoostM2', ...
    'OptimizeHyperparameters', vars, ...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 200, ...
        'ShowPlots', true, ...
        'Verbose', 1));

disp('Optimization complete.');

% =========================================================================
% --- 5. Evaluate the Final, Optimized Model ---
% =========================================================================

disp('Evaluating the final optimized model on the test set...');

% --- Make predictions ---
YPred = predict(Mdl, XTest);

% --- Calculate accuracy ---
accuracy = sum(strcmp(YPred, YTest)) / numel(YTest);

% --- Display the final accuracy ---
fprintf('\nFinal Optimized Boosted Trees Accuracy: %.2f%%\n', accuracy * 100);

% =========================================================================
% --- 6. Visualize Performance with a Confusion Matrix ---
% =========================================================================

disp('Displaying the confusion matrix for the optimized model...');

figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix for Optimized Boosted Trees';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

