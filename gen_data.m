clc;
clear;
close all;
disp("--- Starting Data Generation and Feature Extraction Pipeline ---");

% =========================================================================
% --- 1. Configuration ---
% =========================================================================
model_name = "joint_model";           % Your .slx file name (must be in current folder)
sim_time = 5;                     % Run each simulation for 5 seconds
num_runs_per_scenario = 500;      % 500 simulations for each fault type

% --- Define Output Filenames ---
output_feature_csv_file = 'actuator_feature_data_for_training.csv';
output_raw_mat_file = 'actuator_raw_signal_data.mat';

% =========================================================================
% --- 2. Define Fault Scenarios ---
% =========================================================================
disp('Defining fault scenarios...');
% --- Healthy Case ---
scenarios(1).Label = 'Healthy';
scenarios(1).bearing_friction = 0.001;   
scenarios(1).gear_efficiency = 0.99;     
scenarios(1).sensor_noise_level = 0.001; 

% --- Fault Case 1: Worn Bearing ---
scenarios(2).Label = 'Worn_Bearing';
scenarios(2).bearing_friction = 0.8;   
scenarios(2).gear_efficiency = 0.99;   
scenarios(2).sensor_noise_level = 0.001;

% --- Fault Case 2: Worn Gear (Low Efficiency) ---
scenarios(3).Label = 'Worn_Gear';
scenarios(3).bearing_friction = 0.001;
scenarios(3).gear_efficiency = 0.80;    
scenarios(3).sensor_noise_level = 0.001;

% --- Fault Case 3: Mixed Fault (Worn Bearing + Worn Gear) ---
scenarios(4).Label = 'Mixed_Fault_1';
scenarios(4).bearing_friction = 0.8;   
scenarios(4).gear_efficiency = 0.80;   
scenarios(4).sensor_noise_level = 0.001;

% --- Fault Case 4: Sensor Failure ---
scenarios(5).Label = 'Sensor_Fault';
scenarios(5).bearing_friction = 0.001;
scenarios(5).gear_efficiency = 0.99;   
scenarios(5).sensor_noise_level = 0.1;   % HIGH noise

% =========================================================================
% --- 3. Pre-allocate Data Arrays ---
% =========================================================================
total_runs = length(scenarios) * num_runs_per_scenario;

% --- For the RAW .mat file ---
% Use cell arrays to hold signals of varying lengths
X_raw_signals = cell(total_runs, 1);
Y_labels_raw = cell(total_runs, 1);

% --- For the FEATURE .csv file ---
% Use a standard numeric array for features
X_features = zeros(total_runs, 3); % 3 features
Y_labels_features = cell(total_runs, 1);

data_idx = 1; % Master index for storing data

% =========================================================================
% --- 4. Main Generation Loop ---
% =========================================================================
disp('Loading Simulink model into memory...');
load_system(model_name);
disp('Starting simulation loop...');

for s = 1:length(scenarios)
    
    current_scenario = scenarios(s);
    fprintf('Running Scenario: %s\n', current_scenario.Label);
    
    for run = 1:num_runs_per_scenario
        
        % Print progress update every 50 runs
        if mod(run, 50) == 0
            fprintf('  - Run %d/%d\n', run, num_runs_per_scenario);
        end
        
        % --- 4A. Set Fault Parameters (with randomness) ---
        % (This adds variation to your data, which is good for training)
        bearing_friction = current_scenario.bearing_friction; % * (1 + 0.05*randn); <-- REMOVED
        random_efficiency = current_scenario.gear_efficiency; % * (1 + 0.05*randn); <-- REMOVED
        gear_efficiency = min(1.0, max(0.0, random_efficiency)); 
        sensor_noise_level = current_scenario.sensor_noise_level; % * (1 + 0.05*randn); <-- REMOVED
        
        % --- 4B. Run the Simulation ---
        try
            simIn = Simulink.SimulationInput(model_name);
            simIn = simIn.setVariable('bearing_friction', bearing_friction);
            simIn = simIn.setVariable('gear_efficiency', gear_efficiency);
            simIn = simIn.setVariable('sensor_noise_level', sensor_noise_level);
            % Ensure sim time is set (optional, can be set in model)
            % simIn = simIn.setModelParameter('StopTime', num2str(sim_time)); 
            
            out = sim(simIn);
            
        catch ME
            fprintf('ERROR during simulation run %d: %s\n', run, ME.message);
            fprintf('Skipping this run.\n');
            continue; % Skip this run and move to the next
        end
        
        % --- 4C. Extract RAW Signals ---
        % !! Make sure your Outport blocks are named 'Torque' and 'Vibration'
        input_signal = out.Input.Data;     % <-- ADDED: Extract Input Signal
        torque_signal = out.Torque.Data; 
        vibration_signal = out.Vibration.Data;
        
        % --- 4D. Store RAW Signals (for the .mat file) ---
        % Store all 3 raw signals now
        X_raw_signals{data_idx} = [input_signal(:), torque_signal(:), vibration_signal(:)]; % <-- UPDATED
        Y_labels_raw{data_idx} = current_scenario.Label;
        
        % --- 4E. Extract 3 FEATURES (for the .csv file) ---
        %
        % !! CRITICAL ASSUMPTION !!
        % This script assumes your 3 features are:
        % 1. Mean of Torque
        % 2. RMS of Vibration
        % 3. Kurtosis of Vibration
        % If your original 'merged_data.csv' used different features,
        % you MUST change these 3 lines.
        %
        % --- UPDATED FEATURES based on user feedback ---
        % 1. Mean of Input Signal
        % 2. Mean of Torque Signal
        % 3. Mean of Vibration Signal
        %
        feature_1 = mean(input_signal);
        feature_2 = mean(torque_signal);
        feature_3 = mean(vibration_signal);
        
        % Store features and label for the CSV
        X_features(data_idx, :) = [feature_1, feature_2, feature_3];
        Y_labels_features{data_idx} = current_scenario.Label;
        
        data_idx = data_idx + 1;
    end
end

disp('Simulation loop complete.');

% =========================================================================
% --- 5. Clean and Save Data ---
% =========================================================================

% --- Trim empty pre-allocated rows (if any runs failed) ---
total_completed_runs = data_idx - 1;
if total_completed_runs < total_runs
    disp('Trimming empty rows from failed runs...');
    X_raw_signals = X_raw_signals(1:total_completed_runs);
    Y_labels_raw = Y_labels_raw(1:total_completed_runs);
    
    X_features = X_features(1:total_completed_runs, :);
    Y_labels_features = Y_labels_features(1:total_completed_runs);
end

% --- Convert labels to categorical
Y_labels_raw = categorical(Y_labels_raw);
Y_labels_features = categorical(Y_labels_features);

% --- 5A. Save RAW Signals to .mat file ---
fprintf('Saving RAW signals to %s...\n', output_raw_mat_file);
save(output_raw_mat_file, 'X_raw_signals', 'Y_labels_raw', '-v7.3');
fprintf('Saved %s.\n', output_raw_mat_file);

% --- 5B. Save FEATURES to .csv file for training ---
fprintf('Saving features to %s for training...\n', output_feature_csv_file);

% Create tables for features and labels
T_features = array2table(X_features, 'VariableNames', {'Mean_Input', 'Mean_Torque', 'Mean_Vibration'}); % <-- UPDATED
T_labels = table(Y_labels_features, 'VariableNames', {'Label'});

% Combine into the final table: [Feat1, Feat2, Feat3, Label]
% This matches the format your training script expects
training_table = [T_features, T_labels];

% Write the table to the CSV file
% We set 'WriteVariableNames' to false because your training script
% (readtable) was not expecting a header row. This is a perfect match.
try
    writetable(training_table, output_feature_csv_file, 'WriteVariableNames', false);
    fprintf('Saved %s.\n', output_feature_csv_file);
catch ME
    fprintf('ERROR writing CSV file: %s\n', ME.message);
    disp('Check folder permissions.');
end

disp('--- All Done! ---');
fprintf('You can now run your training script using %s\n', output_feature_csv_file);
