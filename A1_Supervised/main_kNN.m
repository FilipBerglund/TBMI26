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
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf 
                                    % for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors
k = 31;

%% Calculate The Confusion Matrix and the Accuracy
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

acc = 0;
for i = 1:numBins
    bins = 1:numBins;
    bins(i)=[];
    XTrain = combineBins(XBins, bins);
    LTrain = combineBins(LBins, bins);
    XTest  = combineBins(XBins, [i]);
    LTest  = combineBins(LBins, [i]);
    LPredTest  = kNN(XTest , k, XTrain, LTrain);
    % The confucionMatrix
    cM = calcConfusionMatrix(LPredTest, LTest);
    % The accuracy
    acc = acc + calcAccuracy(cM)/numBins;
end
disp(acc)

% Classify training data
LPredTrain = kNN(XTrain, k, XTrain, LTrain);

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
