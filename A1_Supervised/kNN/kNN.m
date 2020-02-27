function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred = zeros(size(X,1),1);

for sample = 1:size(X,1)
    distances = vecnorm(X(sample,:)-XTrain,2,2);
    [~,index] = mink(distances, k);
    index = index';
    
    labelscores = zeros(1,NClasses);
    for i = index
        labelscores(LTrain(i)) = labelscores(LTrain(i)) + 1;
    end
    
    max_index = find(labelscores == max(labelscores));
    if length(max_index) > 1
        for i = index
            for maxi = max_index                
                if LTrain(i) == maxi
                    LPred(sample) = maxi;
                end
            end
        end
    else
        LPred(sample) = max_index;
    end
end
end