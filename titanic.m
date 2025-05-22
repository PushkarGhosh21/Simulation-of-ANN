% Load the Titanic dataset
titanic_train = readtable('C:\Users\Pushkar Ghosh\Downloads\titanic.csv'); % Ensure your CSV file is named 'train.csv'

% Preprocess the data
% Convert categorical variables to numeric
titanic_train.Sex = grp2idx(titanic_train.Sex); % Convert 'male'/'female' to 1/2
titanic_train.Embarked = grp2idx(titanic_train.Embarked); % Convert 'C', 'Q', 'S' to numeric

% Fill missing values (for Age, Fare, and Embarked)
titanic_train.Age = fillmissing(titanic_train.Age, 'constant', median(titanic_train.Age, 'omitnan'));
titanic_train.Fare = fillmissing(titanic_train.Fare, 'constant', median(titanic_train.Fare, 'omitnan'));
titanic_train.Embarked = fillmissing(titanic_train.Embarked, 'constant', mode(titanic_train.Embarked));

% Extract features and target variable
X = [titanic_train.Pclass, titanic_train.Sex, titanic_train.Age, titanic_train.SibSp, ...
     titanic_train.Parch, titanic_train.Fare, titanic_train.Embarked]; % Features
y = titanic_train.Survived; % Target variable

% Normalize features
X = (X - mean(X)) ./ std(X);

% Split the data into training and validation sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
idx = cv.test;

% Separate to training and testing data
XTrain = X(~idx, :);
yTrain = y(~idx);
XTest = X(idx, :);
yTest = y(idx);

% Create and configure the neural network
hiddenLayerSize = 10; % Number of neurons in hidden layer
net = feedforwardnet(hiddenLayerSize);

% Set training parameters
net.trainParam.epochs = 1000; % Number of training epochs
net.trainParam.goal = 0.01;    % Performance goal

% Train the network
[net, tr] = train(net, XTrain', yTrain');

% Test the network
yPredProb = net(XTest');
%yPred = round(yPredProb); % Convert probabilities to binary predictions

% Calculate performance metrics
performance = perform(net, yTest', yPredProb);

% Display results
fprintf('Performance (MSE): %.4f\n', performance);
figure;
plotregression(yTest', yPredProb);
title('Titanic Survival Prediction');