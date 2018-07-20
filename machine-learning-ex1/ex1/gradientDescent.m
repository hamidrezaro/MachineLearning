function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

%% old answer
% for iter = 1:num_iters
%     for i=1:m
%         sigma_part = (X*theta - y).*X(i,:)';
%     end
%         theta_temp = theta - (alpha/m)*sigma_part;
%     
%     theta = theta_temp;
%    
%     J_history(iter) = computeCost(X, y, theta);
% 
% end

%% new answer
for iter = 1:num_iters
    
    theta = theta - (alpha/m)*(X'*(X*theta - y));
    J_history(iter) = computeCost(X, y, theta);
    
end


end
