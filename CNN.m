% --- MATLAB Script for Training a DEEP Neural Network (FCN) ---
%
% This script uses a Fully Connected Network (FCN) from the Deep Learning
% Toolbox, combined with advanced feature engineering, to find complex
% patterns in the data and achieve high accuracy.
%
% =========================================================================
% --- 1. Load and Prepare the Data (from .csv file) ---
% =========================================================================
% --- Specify the file path ---
filePath = 'C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\Scripts\actuator_feature_data_for_training.csv'; % <-- CHANGED TO .csv

% --- Load the data from the .csv file ---
disp('Step 1: Loading and preparing data from .csv file...');
try
    % Use readtable, and specify that the data has NO header
    % 'readtable' will auto-assign names Var1, Var2, Var3, Var4
    dataTable = readtable(filePath, 'ReadVariableNames', false);
catch ME
    error('Failed to load the .csv file. Check the path. Original error: %s', ME.message);
end

% --- Separate predictors (features) and response (labels) ---
% UPDATED based on your CSV image (Features in 1-3, Label in 4)
X_original = table2array(dataTable(:, 1:3)); % Features are now columns 1, 2, 3
Y = categorical(table2array(dataTable(:, 4))); % Labels are now in column 4

totalSamples = size(X_original, 1);
numClasses = numel(categories(Y));
fprintf('Loaded %d samples. Found %d classes.\n', totalSamples, numClasses);

% =========================================================================
% --- 2. ADVANCED Feature Engineering (UPDATED FOR 3 FEATURES) ---
% =========================================================================
disp('Step 2: Performing ADVANCED feature engineering...');

X1 = X_original(:, 1); % Feature 1
X2 = X_original(:, 2); % Feature 2
X3 = X_original(:, 3); % Feature 3

% --- Create a new, rich set of features (based on 3 inputs) ---
feat_inter_12 = X1 .* X2;
feat_inter_13 = X1 .* X3;
feat_inter_23 = X2 .* X3;
feat_sq_1 = X1.^2;
feat_sq_2 = X2.^2;
feat_sq_3 = X3.^2;
feat_ratio_12 = X2 ./ (X1 + 1e-6);
feat_ratio_13 = X3 ./ (X1 + 1e-6);
feat_ratio_23 = X3 ./ (X2 + 1e-6);
feat_sum_12 = X1 + X2;
feat_sum_13 = X1 + X3;
feat_sum_23 = X2 + X3;
feat_diff_12 = X1 - X2;
feat_diff_13 = X1 - X3;
feat_diff_23 = X2 - X3;

% --- Combine original features with all new engineered features ---
X_features = [X_original, ...
     feat_inter_12, feat_inter_13, feat_inter_23, ...
     feat_sq_1, feat_sq_2, feat_sq_3, ...
     feat_ratio_12, feat_ratio_13, feat_ratio_23, ...
     feat_sum_12, feat_sum_13, feat_sum_23, ...
     feat_diff_12, feat_diff_13, feat_diff_23];

numFeatures = size(X_features, 2);
fprintf('New, richer feature set created with %d total features.\n', numFeatures);

% =========================================================================
% --- 3. Shuffle, Split, and Normalize Data ---
% =========================================================================
disp('Step 3: Shuffling, splitting, and normalizing...');

% --- Create a random partition ---
cv = cvpartition(totalSamples, 'HoldOut', 0.20);
idx = cv.test;

% --- Create the final training and test sets ---
XTrain = X_features(~idx, :);
YTrain = Y(~idx, :);
XTest  = X_features(idx, :);
YTest  = Y(idx, :);

fprintf('Training set size: %d samples\n', size(XTrain, 1));
fprintf('Testing set size: %d samples\n', size(XTest, 1));

% --- Normalize Features (z-score) ---
% This is CRITICAL for deep learning
disp('Normalizing features...');
mu = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);

XTrain = (XTrain - mu) ./ sigma;
XTest  = (XTest - mu) ./ sigma;

% Handle any potential NaN values
XTrain(isnan(XTrain)) = 0;
XTest(isnan(XTest)) = 0;

% =========================================================================
% --- 4. Define the DEEP Neural Network Architecture (NEW) ---
% =========================================================================
disp('Step 4: Defining a DEEP Fully Connected Network architecture...');

layers = [
    % This layer is for tabular feature data, not images
    featureInputLayer(numFeatures, 'Name', 'input') 
    
    % --- Block 1 ---
    fullyConnectedLayer(256, 'Name', 'fc_1')
    batchNormalizationLayer('Name', 'batchnorm_1')
    reluLayer('Name', 'relu_1')
    dropoutLayer(0.3, 'Name', 'dropout_1') % Add dropout for regularization
    
    % --- Block 2 ---
    fullyConnectedLayer(128, 'Name', 'fc_2')
    batchNormalizationLayer('Name', 'batchnorm_2')
    reluLayer('Name', 'relu_2')
    dropoutLayer(0.3, 'Name', 'dropout_2')
    
    % --- Block 3 ---
    fullyConnectedLayer(64, 'Name', 'fc_3')
    batchNormalizationLayer('Name', 'batchnorm_3')
    reluLayer('Name', 'relu_3')
    dropoutLayer(0.2, 'Name', 'dropout_3')
    
    % --- Output Head ---
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% analyzeNetwork(layers); % Uncomment to visualize the network

% =========================================================================
% --- 5. Specify Training Options ---
% =========================================================================
disp('Step 5: Specifying training options with Learning Rate Schedule...');
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ... % No reshape needed
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...    % Use learning rate schedule
    'LearnRateDropPeriod', 40, ...         % Drop the rate every 40 epochs
    'LearnRateDropFactor', 0.2);           % Drop it by a factor of 0.2

% =========================================================================
% --- 6. Train the Model ---
% =========================================================================
disp('Step 6: Training the model... (This will open a training progress window)');
net = trainNetwork(XTrain, YTrain, layers, options); % No reshape needed

% =========================================================================
% --- 7. Evaluate the Model ---
% =========================================================================
disp('Step 7: Evaluating the trained model on the test set...');

% Classify the test data
predictedLabels = classify(net, XTest); % No reshape needed

% Calculate accuracy
accuracy = sum(predictedLabels == YTest) / numel(YTest);
fprintf('\nModel Accuracy on the Test Set: %.2f%%\n', accuracy * 100);

% Display a confusion matrix
figure;
confusionchart(YTest, predictedLabels);
title('Confusion Matrix for Deep Network (Test Data)');
% --- 8. Save the Model and Preprocessing Parameters ---
disp('Step 8: Saving model and normalization parameters...');

% Save everything needed for future predictions into one file
save('MyDeepModel.mat', 'net', 'mu', 'sigma');

disp('Model saved as MyDeepModel.mat');
