function [J grad] = nnCostFunction(nn_params,
                                   input_layer_size,
                                   hidden_layer_size, 
                                   num_labels,
                                   X, y, lambda)
%NNCOSTFUNCTION la cong cu dung de tinh toan cost function cua neural network cho 2 hay nhieu layer
%nn-params la dieu can thiet de hoi tu cac diem lai nho cac weight 
%reshape cho theta1 va theta2, weight
%cho 2 layer trong nn
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
                 num_labels, (hidden_layer_size + 1));

% thiet lap chieu ma tran
m = size(X,1);
% chung ta can tra ve gia tri dung voi cac bien da tao
J = 0
% tao ra matrix so 0 tuong ung voi so chieu cua theta 1 va theta 2
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



% recode y to Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% feedforward
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

% calculte penalty
p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

% calculate J
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

% calculate sigmas
sigma3 = a3.-Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

% accumulate gradients
delta_1 = (sigma2'*a1);
delta_2 = (sigma3'*a2);

% calculate regularized gradient
p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
