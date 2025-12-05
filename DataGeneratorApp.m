classdef DataGeneratorApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                     matlab.ui.Figure
        MainGridLayout               matlab.ui.container.GridLayout
        LeftPanel                    matlab.ui.container.Panel
        SelectFaultScenarioLabel     matlab.ui.control.Label
        ScenarioDropDown             matlab.ui.control.DropDown
        GeneratePredictButton        matlab.ui.control.Button
        StatusLampLabel              matlab.ui.control.Label
        StatusLamp                   matlab.ui.control.Lamp
        StatusLabel                  matlab.ui.control.Label
        RightPanel                   matlab.ui.container.Panel
        PredictionResultLabel        matlab.ui.control.Label
        ResultLabel                  matlab.ui.control.Label
        SignalPlotsAxes              matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        % --- Properties to hold our Simulink model and AI model ---
        model_name = "joint_model"; % <-- CHANGED: Use model name, NOT full path
        sim_time = 5;               % Simulation time
        
        % --- AI Model Properties (loaded at startup) ---
        trainedNet  % Holds the 'net' from MyDeepModel.mat
        norm_mu     % Holds the 'mu' from MyDeepModel.mat
        norm_sigma  % Holds the 'sigma' from MyDeepModel.mat
    end
    
    
    methods (Access = private)
        
        % Code that executes at app startup
        function startupFcn(app)
            disp('App starting up...');
            app.StatusLabel.Text = 'Loading AI Model...';
            app.StatusLamp.Color = 'yellow';
            
            try
                % Load the AI model ONCE at startup
                ai_model_data = load('MyDeepModel.mat');
                app.trainedNet = ai_model_data.net;
                app.norm_mu    = ai_model_data.mu;
                app.norm_sigma = ai_model_data.sigma;
                
                app.StatusLabel.Text = 'Ready. Select a scenario.';
                app.StatusLamp.Color = 'green';
                disp('AI Model loaded successfully.');
            catch ME
                app.StatusLabel.Text = 'Error: MyDeepModel.mat not found!';
                app.StatusLamp.Color = 'red';
                app.GeneratePredictButton.Enable = 'off';
                disp('Error loading AI model:');
                disp(ME.message);
            end
            
            % Check if Simulink model exists IN THE CURRENT FOLDER
            % We check for the file name with the .slx extension
            if ~isfile(app.model_name + ".slx")
                app.StatusLabel.Text = 'Error: joint_model.slx not in current folder!'; % <-- CHANGED
                app.StatusLamp.Color = 'red';
                app.GeneratePredictButton.Enable = 'off';
                disp('Error: Simulink model not found in current folder.');
            end
        end

        % Button pushed function: GeneratePredictButton
        function GeneratePredictButtonPushed(app, event)
            app.GeneratePredictButton.Enable = 'off'; % Disable button
            app.StatusLamp.Color = 'yellow';
            
            % 1. Get selected fault from dropdown
            scenarioLabel = app.ScenarioDropDown.Value;
            app.StatusLabel.Text = ['Setting up ' scenarioLabel '...'];
            drawnow; % Force UI update
            
            % 2. Define fault parameters
            switch scenarioLabel
                case 'Healthy'
                    bearing_friction = 0.001;
                    gear_efficiency = 0.99;
                    sensor_noise_level = 0.001;
                case 'Worn Bearing'
                    bearing_friction = 0.99;
                    gear_efficiency = 0.99;
                    sensor_noise_level = 0.001;
                case 'Worn Gear'
                    bearing_friction = 0.001;
                    gear_efficiency = 0.1;
                    sensor_noise_level = 0.001;
                case 'Mixed Fault'
                    bearing_friction = 0.99;
                    gear_efficiency = 0.1;
                    sensor_noise_level = 0.001;
                case 'Sensor Fault'
                    bearing_friction = 0.001;
                    gear_efficiency = 0.99;
                    sensor_noise_level = 0.5;
                otherwise
                    bearing_friction = 0.001;
                    gear_efficiency = 0.99;
                    sensor_noise_level = 0.001;
            end
            
            % 3. Run the Simulink Simulation
            app.StatusLabel.Text = 'Running Simulink model...';
            try
                % Load the system into memory first
                load_system(app.model_name); % <-- ADDED
                
                % Use a SimulationInput object to pass variables
                simIn = Simulink.SimulationInput(app.model_name); % This now correctly uses "joint_model"
                simIn = simIn.setVariable('bearing_friction', bearing_friction);
                simIn = simIn.setVariable('gear_efficiency', gear_efficiency);
                simIn = simIn.setVariable('sensor_noise_level', sensor_noise_level);
                
                % NOTE: We are removing this line, as sim_time is set in the model
                % simIn = simIn.setExternalInput('sim_time', app.sim_time);
                
                % Run the simulation
                out = sim(simIn);
                
                % 4. Extract RAW Signals
                app.StatusLabel.Text = 'Extracting signals...';
                torque_signal = out.Torque.Data;
                vibration_signal = out.Vibration.Data;
                time_vector = out.Torque.Time;
                
                % 5. Plot RAW Signals in the App
                plot(app.SignalPlotsAxes, time_vector, torque_signal, 'b-', 'LineWidth', 1.5);
                hold(app.SignalPlotsAxes, 'on');
                plot(app.SignalPlotsAxes, time_vector, vibration_signal, 'r--', 'LineWidth', 1.5);
                hold(app.SignalPlotsAxes, 'off');
                legend(app.SignalPlotsAxes, 'Torque', 'Vibration', 'Location', 'northeast');
                title(app.SignalPlotsAxes, ['Raw Signals: ' scenarioLabel]);
                xlabel(app.SignalPlotsAxes, 'Time (s)');
                ylabel(app.SignalPlotsAxes, 'Amplitude');
                grid(app.SignalPlotsAxes, 'on');

            catch ME
                app.StatusLabel.Text = 'Error during simulation.';
                app.StatusLamp.Color = 'red';
                app.GeneratePredictButton.Enable = 'on';
                disp('Simulation Error:');
                disp(ME.message);
                return; % Stop execution
            end
            
            % 6. Apply Feature Extraction (The "Missing Link")
            app.StatusLabel.Text = 'Calculating features...';
            
            % !! This is the "guess" from our previous conversation !!
            % F1: Mean Torque, F2: RMS Vibration, F3: Kurtosis Vibration
            try
                f1 = mean(torque_signal);
                f2 = rms(vibration_signal);
                f3 = kurtosis(vibration_signal);
                X_original = [f1, f2, f3];
            catch FE_ME
                app.StatusLabel.Text = 'Error calculating features from signal.';
                app.StatusLamp.Color = 'red';
                app.GeneratePredictButton.Enable = 'on';
                disp('Feature Extraction Error:');
                disp(FE_ME.message);
                return;
            end

            % 7. Apply Feature Engineering (3 -> 18 features)
            app.StatusLabel.Text = 'Engineering features...';
            X1 = X_original(:, 1); X2 = X_original(:, 2); X3 = X_original(:, 3);
            feat_inter_12 = X1 .* X2; feat_inter_13 = X1 .* X3; feat_inter_23 = X2 .* X3;
            feat_sq_1 = X1.^2; feat_sq_2 = X2.^2; feat_sq_3 = X3.^2;
            feat_ratio_12 = X2 ./ (X1 + 1e-6); feat_ratio_13 = X3 ./ (X1 + 1e-6); feat_ratio_23 = X3 ./ (X2 + 1e-6);
            feat_sum_12 = X1 + X2; feat_sum_13 = X1 + X3; feat_sum_23 = X2 + X3;
            feat_diff_12 = X1 - X2; feat_diff_13 = X1 - X3; feat_diff_23 = X2 - X3;

            X_features = [X_original, ...
                 feat_inter_12, feat_inter_13, feat_inter_23, ...
                 feat_sq_1, feat_sq_2, feat_sq_3, ...
                 feat_ratio_12, feat_ratio_13, feat_ratio_23, ...
                 feat_sum_12, feat_sum_13, feat_sum_23, ...
                 feat_diff_12, feat_diff_13, feat_diff_23];
            
            % 8. Normalize using loaded mu and sigma
            app.StatusLabel.Text = 'Normalizing features...';
            X_final = (X_features - app.norm_mu) ./ app.norm_sigma;
            X_final(isnan(X_final)) = 0; % Handle NaNs
            
            % 9. Predict using the AI Model
            app.StatusLabel.Text = 'Predicting...';
            try
                predictedLabel = classify(app.trainedNet, X_final);
                
                % 10. Display Result
                app.ResultLabel.Text = string(predictedLabel);
                
                % Update font color based on prediction
                if predictedLabel == 'Healthy'
                    app.ResultLabel.FontColor = [0, 0.6, 0.2]; % Dark Green
                else
                    app.ResultLabel.FontColor = [0.8, 0, 0]; % Dark Red
                end
                
                app.StatusLabel.Text = 'Done. Ready for next run.';
                app.StatusLamp.Color = 'green';
                
            catch PRED_ME
                app.StatusLabel.Text = 'Error during prediction.';
                app.StatusLamp.Color = 'red';
                disp('Prediction Error:');
                disp(PRED_ME.message);
            end
            
            app.GeneratePredictButton.Enable = 'on'; % Re-enable button
        end
    end
    

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 863 479];
            app.UIFigure.Name = 'Simulink Data Generator & Validator';

            % Create MainGridLayout
            app.MainGridLayout = uigridlayout(app.UIFigure);
            app.MainGridLayout.ColumnWidth = {300, '1x'};
            app.MainGridLayout.RowHeight = {'1x'};

            % Create LeftPanel
            app.LeftPanel = uipanel(app.MainGridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create SelectFaultScenarioLabel
            app.SelectFaultScenarioLabel = uilabel(app.LeftPanel);
            app.SelectFaultScenarioLabel.FontSize = 14;
            app.SelectFaultScenarioLabel.FontWeight = 'bold';
            app.SelectFaultScenarioLabel.Position = [31 391 161 22];
            app.SelectFaultScenarioLabel.Text = 'Select Fault Scenario';

            % Create ScenarioDropDown
            app.ScenarioDropDown = uidropdown(app.LeftPanel);
            app.ScenarioDropDown.Items = {'Healthy', 'Worn Bearing', 'Worn Gear', 'Mixed Fault', 'Sensor Fault'};
            app.ScenarioDropDown.FontSize = 14;
            app.ScenarioDropDown.Position = [31 360 238 27];
            app.ScenarioDropDown.Value = 'Healthy';

            % Create GeneratePredictButton
            app.GeneratePredictButton = uibutton(app.LeftPanel, 'push');
            app.GeneratePredictButton.ButtonPushedFcn = createCallbackFcn(app, @GeneratePredictButtonPushed, true);
            app.GeneratePredictButton.BackgroundColor = [0.0745 0.6235 1];
            app.GeneratePredictButton.FontSize = 16;
            app.GeneratePredictButton.FontColor = [1 1 1];
            app.GeneratePredictButton.Position = [31 292 238 41];
            app.GeneratePredictButton.Text = 'Generate & Predict';

            % Create StatusLampLabel
            app.StatusLampLabel = uilabel(app.LeftPanel);
            app.StatusLampLabel.HorizontalAlignment = 'right';
            app.StatusLampLabel.FontSize = 14;
            app.StatusLampLabel.Position = [47 72 50 22];
            app.StatusLampLabel.Text = 'Status:';

            % Create StatusLamp
            app.StatusLamp = uilamp(app.LeftPanel);
            app.StatusLamp.Position = [108 72 20 20];

            % Create StatusLabel
            app.StatusLabel = uilabel(app.LeftPanel);
            app.StatusLabel.FontSize = 14;
            app.StatusLabel.Position = [47 38 222 35];
            app.StatusLabel.Text = 'Initializing...';

            % Create RightPanel
            app.RightPanel = uipanel(app.MainGridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create PredictionResultLabel
            app.PredictionResultLabel = uilabel(app.RightPanel);
            app.PredictionResultLabel.FontSize = 18;
            app.PredictionResultLabel.FontWeight = 'bold';
            app.PredictionResultLabel.Position = [32 426 150 24];
            app.PredictionResultLabel.Text = 'Prediction Result:';

            % Create ResultLabel
            app.ResultLabel = uilabel(app.RightPanel);
            app.ResultLabel.FontSize = 36;
            app.ResultLabel.FontWeight = 'bold';
            app.ResultLabel.Position = [190 411 340 49];
            app.ResultLabel.Text = '---';

            % Create SignalPlotsAxes
            app.SignalPlotsAxes = uiaxes(app.RightPanel);
            title(app.SignalPlotsAxes, 'Raw Signals (Torque & Vibration)')
            xlabel(app.SignalPlotsAxes, 'Time (s)')
            ylabel(app.SignalPlotsAxes, 'Amplitude')
            zlabel(app.SignalPlotsAxes, 'Z')
            app.SignalPlotsAxes.Position = [20 38 521 342];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = DataGeneratorApp

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end