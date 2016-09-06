function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
l=numHidden+2;
m=numel(labels);
hAct = cell(l, 1);
hAct{1}=data;
gradStack = cell(l-1, 1);

%% forward prop
for i=1:numHidden
    hAct{i+1}=sigmoid(bsxfun(@plus, stack{i}.W*hAct{i}, stack{i}.b));
end
E=exp(bsxfun(@plus, stack{l-1}.W*hAct{l-1}, stack{l-1}.b));
hAct{l}=bsxfun(@rdivide,E,sum(E));
H=hAct{l};
pred_prob=H;

%% return here if only predictions desired.
if po
    cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
    grad = [];  
    return;
end
    
%% compute cost
I=sub2ind(size(H),labels',1:size(H,2));
ceCost=-sum(log(H(I)));

%% compute gradients using backpropagation
e=zeros(size(H));
e(I)=1;
e=H-e;
for j=l-1:-1:1
    a=hAct{j};
    gradStack{j}.W=e*a'/m;
    gradStack{j}.b=mean(e,2);
    e=stack{j}.W'*e.*a.*(1-a);
end

%% compute weight penalty cost and gradient for non-bias terms
wCost=0;
%{
for j=1:l-1
    wCost=wCost+sum(sum(stack{j}.W.^2));
end
wCost=wCost*ei.lambda/2;
%}
cost=ceCost+wCost;
%{
for j=1:l-1
    gradStack{j}.W=gradStack{j}.W+ei.lambda*stack{j}.W;
end
%}
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end