%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 5;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf;  % Number of samples per label per bin, set to inf 
                                    % for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
max_k = 1000;
acc = zeros(1,max_k);
for i = 1:numBins
    bins = 1:numBins;
    bins(i)=[];
    XTrain = combineBins(XBins, bins);
    LTrain = combineBins(LBins, bins);
    XTest  = combineBins(XBins, [i]);
    LTest  = combineBins(LBins, [i]);

    % Set the number of neighbors

    for k = max_k:max_k
        LPredTest  = kNN(XTest , k, XTrain, LTrain);

        % The confucionMatrix
        cM = calcConfusionMatrix(LPredTest, LTest);

        % The accuracy
        acc(k) = acc(k) + calcAccuracy(cM)/numBins;
    end
end
plot(1:1,acc,'*')