function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


%% calculating J
j_by_m = 0;
reg = 0;
for i = 1:m
    z = X(i,:)*theta;
    j_by_m = j_by_m + y(i)*log(sigmoid(z)) + (1-y(i))*log(1 - sigmoid(z));
end

for j = 2:n
    
    reg = reg + theta(j)^2;
    
end

J = (-(1/m)*j_by_m) + (1/(2*m))*lambda*reg;

%% end of calculating J


%% calculating Gradients
for j = 1:n
    
    grad_by_m = 0;
    
    for i = 1:m
        z = X(i,:)*theta;
        grad_by_m = grad_by_m + (sigmoid(z) - y(i))*X(i,j) + (j~=1)*lambda*(1/m)*theta(j);        
    end
    grad(j) = (1/m)*grad_by_m;
    
end

%% end ofcalculating Gradients


end
