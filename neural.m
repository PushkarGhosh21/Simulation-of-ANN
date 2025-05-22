num_inputs = 3;
num_hidden = 10;
num_output = 1;

activation_fun = 'sigmoid';

weights = rand(num_inputs, num_hidden);
bias = zeros(1, num_hidden);

X = [0 0 1; 1 1 1; 1 0 1; 0 1 1];
y = [0; 1; 1; 0];

algo = 'backpropagation';
learning_rate = 0.1;
num_iterations = 100;

sigmoid = @(x) 1./(1+exp(-x));
sigmoid_derivative = @(x) sigmoid(x) .* (1 - sigmoid(x));

for i = 1:num_iterations
    net = X * weights + bias;
    op = sigmoid(net); %op = output
    error = op - y;
    delta = error .* sigmoid_derivative(net);  %i use backpropagration algo
    weights = weights - learning_rate .* (X' * delta);
    bias = bias - learning_rate .* sum(delta, 1); %adjusting bias
end

X_test = [0 1 0; 1 0 1];
output = sigmoid(X_test * weights + bias);

%Mean squared error
mse = mean((output - y(1:size(X_test,1))).^2);
fprintf('Mean squared error: %f\n', mse);

figure;
plot(X(:, 1), X(:, 2), 'bo');
hold on;
plot(y, 'ro');
plot(output, 'go');
title('Artificial Neural Network Simulation');
xlabel('Input1');
ylabel('Input2');
legend('Input Data', 'Target Outputs', 'Predicted Outputs');