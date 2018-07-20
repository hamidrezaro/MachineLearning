function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
[m, n] = size(X); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


j_by_m = 0;
for i = 1:m
    z = X(i,:)*theta;
    j_by_m = j_by_m + y(i)*log(sigmoid(z)) + (1-y(i))*log(1 - sigmoid(z));
end

J = -(1/m)*j_by_m;

for j = 1:n
    
    grad_by_m = 0;
    
    for i = 1:m
        
        z = X(i,:)*theta;
        grad_by_m = grad_by_m + (sigmoid(z) - y(i))*X(i,j);
        
    end
    
    grad(j) = (1/m)*grad_by_m;
    
end



% =============================================================

end
