function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

normalTerm=0;
for i=1:m,
	z = X(i, :) * theta;
	g = sigmoid(z);
	normalTerm = normalTerm + (-y(i) * log(g) - (1 - y(i)) * log( 1 - g ));
end;

regularizationTerm=0;
for i=2:size(theta,1),
	regularizationTerm = regularizationTerm + theta(i)*theta(i);
end;

J = (normalTerm+lambda/2*regularizationTerm)/m;

for j=1:size(theta,1),
  grad(j) = 0;
  normalTerm=0;
  for i=1:m,
	z = X(i, :) * theta;
	g = sigmoid(z);
	normalTerm = normalTerm + (g - y(i)) * X(i,j);
  end;
  if j>=2,
	grad(j) = (normalTerm+ lambda*theta(j))/ m;
  else
	grad(j) = normalTerm/ m;
  end;
end;

% =============================================================

end
