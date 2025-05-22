% Load the Boston Housing dataset
data = readtable('C:\Users\Pushkar Ghosh\Downloads\BostonHousing.csv'); % Ensure your CSV file is named 'housing.csv'

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
net.trainParam.epochs = 1000; % Number of training epochs
net.trainParam.goal = 0.001;   % Performance goal


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