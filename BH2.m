% Load the Boston Housing dataset
data = readtable('C:\Users\Pushkar Ghosh\Downloads\BostonHousing.csv'); % Ensure your CSV file is named 'PBostonHousing.csv'

% Extract features and target variable
X = data{:, 1:end-1}; % Features (all columns except the last)
y = data{:, end};     % Target variable (last column)

% Normalize the features
X = (X - mean(X)) ./ std(X);

% Split the data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
idx = cv.test;

% Separate to training and testing data
XTrain = X(~idx, :);
yTrain = y(~idx);
XTest = X(idx, :);
yTest = y(idx);

% Create and configure the neural network
hiddenLayerSize = 10; % Number of neurons in hidden layer
net = fitnet(hiddenLayerSize);

% Set training parameters
net.trainParam.epochs = 1000; % Increase number of training epochs
net.trainParam.goal = 0.01;    % Performance goal
net.trainParam.min_grad = 1e-6; % Minimum gradient for convergence
net.trainParam.max_fail = 10;   % Maximum validation failures before stopping

% Train the network
[net, tr] = train(net, XTrain', yTrain');

% Test the network
yPred = net(XTest');

% Calculate performance metrics
performance = perform(net, yTest', yPred);

% Display results
fprintf('Performance (MSE): %.4f\n', performance);
figure;
plotregression(yTest', yPred);
title('Regression of Predicted vs Actual Prices');

% Calculate R-squared for better accuracy measure
SS_res = sum((yTest' - yPred).^2); % Residual sum of squares
SS_tot = sum((yTest' - mean(yTest')).^2); % Total sum of squares
R_squared = 1 - (SS_res / SS_tot); % R-squared value

fprintf('R-squared: %.4f\n', R_squared);
% Calculate a form of accuracy for regression predictions
tolerance_percentage = 0.1; % Define a tolerance of 10%
tolerance_value = tolerance_percentage * mean(yTest); 

accuracy_count = sum(abs(yPred' - yTest) <= tolerance_value);
accuracy_percentage = (accuracy_count / length(yTest)) * 120;

fprintf('Accuracy (within %.0f%% of actual values): %.2f%%\n', tolerance_percentage*100, accuracy_percentage);