% --- MATLAB Script for Training an OPTIMIZED Neural Network Model ---
%
% This script uses a Neural Network (fitcnet) with Bayesian optimization
% to automatically tune hyperparameters for the highest possible accuracy.
%
% =========================================================================
% --- 1. Load and Prepare the Data (from .mat file) ---
% =========================================================================

% --- Specify the file path (UPDATED) ---
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
% We will still create these features, as they provide useful inputs
% for the neural network to learn from.
disp('Performing ADVANCED feature engineering for 4 input features...');

X1 = X_original(:, 1); % Torque_RMS
X2 = X_original(:, 2); % Torque_Mean
X3 = X_original(:, 3); % Vibration_Std
X4 = X_original(:, 4); % Vibration_Max

% --- Create a new, rich set of features ---
feat_inter_12 = X1 .* X2;
feat_inter_13 = X1 .* X3;
feat_inter_14 = X1 .* X4;
feat_inter_23 = X2 .* X3;
feat_inter_24 = X2 .* X4;
feat_inter_34 = X3 .* X4;
feat_sq_1 = X1.^2;
feat_sq_2 = X2.^2;
feat_sq_3 = X3.^2;
feat_sq_4 = X4.^2;
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
% --- 3b. Normalize Features (CORRECTED) ---
% =========================================================================
% Neural Networks require normalized features (mean 0, std dev 1).
% We must calculate the mean/std from the TRAINING set only.
disp('Normalizing features...');
% --- Calculate mu (mean) and sigma (std dev) from the training data ---
mu = mean(XTrain, 1);    % 1 = mean of each column, returns a 1xM row vector
sigma = std(XTrain, 0, 1); % 0 = default normalization, 1 = std dev of each column

% Apply the normalization to both training and test sets
% MATLAB will automatically subtract/divide the 1xM vector from/by each row
XTrain = (XTrain - mu) ./ sigma;
XTest  = (XTest - mu) ./ sigma;

% Handle any potential NaN values if a column had zero std dev
XTrain(isnan(XTrain)) = 0;
XTest(isnan(XTest)) = 0;


% =========================================================================
% --- 4. Find Best Hyperparameters via Optimization (Using NEURAL NETWORK) ---
% =========================================================================
% To get high accuracy, we automatically search for the best model settings.
disp('Starting hyperparameter optimization for Neural Network (fitcnet)...');

% --- Define the hyperparameters (STABLE CONFIGURATION - FINAL) ---
% We fix NumLayers to 3 and optimize the sizes of those 3 layers.
% We must define all 5 layer sizes for the optimizer to be stable.
vars = [optimizableVariable('Activations', {'relu', 'tanh', 'sigmoid'}, 'Type', 'categorical', 'Optimize', true);
        optimizableVariable('Lambda', [1e-5, 1], 'Type', 'real', 'Transform', 'log', 'Optimize', true);
        optimizableVariable('Layer_1_Size', [10, 100], 'Type', 'integer', 'Optimize', true);
        optimizableVariable('Layer_2_Size', [10, 100], 'Type', 'integer', 'Optimize', true);
        optimizableVariable('Layer_3_Size', [10, 100], 'Type', 'integer', 'Optimize', true);
        optimizableVariable('Layer_4_Size', [10, 100], 'Type', 'integer', 'Optimize', false); % Do not optimize
        optimizableVariable('Layer_5_Size', [10, 100], 'Type', 'integer', 'Optimize', false)]; % Do not optimize

% --- Run Bayesian Optimization with fitcnet ---
% We pass 'NumLayers' as a fixed argument, and 'vars' to optimize.
Mdl = fitcnet(XTrain, YTrain, ...
    'NumLayers', 3, ... % Fix to 3 layers
    'OptimizeHyperparameters', vars, ...
    'Standardize', false, ... % Tell the model not to standardize (we already did)
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 60, ...
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
fprintf('\nFinal Optimized Neural Network Accuracy: %.2f%%\n', accuracy * 100);

% =========================================================================
% --- 6. Visualize Performance with a Confusion Matrix ---
% =========================================================================

disp('Displaying the confusion matrix for the optimized model...');

figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix for Optimized Neural Network';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

