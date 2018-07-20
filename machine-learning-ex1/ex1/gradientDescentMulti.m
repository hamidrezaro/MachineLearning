function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
J_history = zeros(num_iters, 1);

%% old answer
% for iter = 1:num_iters
%     
%     temp_theta = zeros(n,1);
%     
%     for j=1:n
%         
%         pd_ofJ = 0;
%         for i=1:m
%             pd_ofJ = pd_ofJ + ( theta' * (X(i,:))' - y(i) )*X(i,j);            
%         end
%         
%         temp_theta(j, 1) = theta(j, 1) - alpha*pd_ofJ/m;
%         
%     end
% 
%     theta = temp_theta;
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
