% MATLAB Script for Automating FAULT SIGNAL Data Generation
clc;
clear;
close all;
disp("Starting data generation for RAW SIGNALS...");

% --- 1. Configuration ---
model_name = "C:\Users\Arnav Oberoi\Documents\MATLAB\AI Project\joint_model.slx"; % Your .slx file name
sim_time = 5;                     % Run each simulation for 5 seconds
num_runs_per_scenario = 500; % Number of simulations for each fault type

% --- 2. Define Fault Scenarios ---
% These variable names MUST match the variables in your Simscape blocks.

% --- Healthy Case ---
scenarios(1).Label = 'Healthy';
scenarios(1).bearing_friction = 0.001;   % Healthy friction
scenarios(1).gear_efficiency = 0.99;     % Healthy efficiency (e.g., 99%)
scenarios(1).sensor_noise_level = 0.001; % Base sensor noise

% --- Fault Case 1: Worn Bearing ---
scenarios(2).Label = 'Worn_Bearing';
scenarios(2).bearing_friction = 0.8;   % HIGH friction
scenarios(2).gear_efficiency = 0.99;   % Healthy efficiency
scenarios(2).sensor_noise_level = 0.001;

% --- Fault Case 2: Worn Gear (Low Efficiency) ---
scenarios(3).Label = 'Worn_Gear';
scenarios(3).bearing_friction = 0.001;
scenarios(3).gear_efficiency = 0.80;    % FAULTY (low) efficiency (e.g., 80%)
scenarios(3).sensor_noise_level = 0.001;

% --- Fault Case 3: Mixed Fault (Worn Bearing + Worn Gear) ---
scenarios(4).Label = 'Mixed_Fault_1';
scenarios(4).bearing_friction = 0.8;   % HIGH friction
scenarios(4).gear_efficiency = 0.80;   % FAULTY (low) efficiency
scenarios(4).sensor_noise_level = 0.001;

% --- Fault Case 4: Sensor Failure ---
scenarios(5).Label = 'Sensor_Fault';
scenarios(5).bearing_friction = 0.001;
scenarios(5).gear_efficiency = 0.99;   % Healthy efficiency
scenarios(5).sensor_noise_level = 0.1;   % HIGH noise

% --- Pre-allocate cell arrays for the raw signal data ---
total_runs = length(scenarios) * num_runs_per_scenario;
X_raw_signals = cell(total_runs, 1);
Y_labels = cell(total_runs, 1);
data_idx = 1; % Index for storing data

% --- 3. Main Generation Loop ---
for s = 1:length(scenarios)
    
    current_scenario = scenarios(s);
    fprintf('Running Scenario: %s\n', current_scenario.Label);
    
    for run = 1:num_runs_per_scenario
        
        % --- 3A. Set Fault Parameters (with a little randomness) ---
        bearing_friction = current_scenario.bearing_friction * (1 + 0.05*randn);
        random_efficiency = current_scenario.gear_efficiency * (1 + 0.05*randn); 
        gear_efficiency = min(1.0, max(0.0, random_efficiency)); 
        sensor_noise_level = current_scenario.sensor_noise_level * (1 + 0.05*randn);
        
        % --- 3B. Run the Simulation ---
        fprintf('  - Run %d/%d\n', run, num_runs_per_scenario);
        out = sim(model_name, sim_time);
        
        % --- 3C. Extract RAW Signals ---
        % !! Make sure your Outport blocks are named 'Torque' and 'Vibration'
        torque_signal = out.Torque.Data; 
        vibration_signal = out.Vibration.Data;
        
        % --- 3D. Store RAW Signals and Label ---
        % Ensure signals are column vectors and combine them as channels
        % The result is a single [NumTimeSteps x 2] matrix
        combined_signal = [torque_signal(:), vibration_signal(:)];
        
        % Store the signal and label
        X_raw_signals{data_idx} = combined_signal;
        Y_labels{data_idx} = current_scenario.Label;
        
        data_idx = data_idx + 1;
    end
end

% --- 4. Finalize and Save ---
% Convert labels to a categorical array
Y_labels = categorical(Y_labels);

% Save the final dataset to a new .mat file
new_filename = 'actuator_raw_signal_data.mat';
save(new_filename, 'X_raw_signals', 'Y_labels');

disp('----------------------------------');
disp('Data generation complete!');
fprintf('Your new raw signal data is saved in %s\n', new_filename);
