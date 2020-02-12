%% Hyper-parameters
clear
clc

% Number of randomized Haar-features
nbrHaarFeatures = 500;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 300;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:min(25,nbrHaarFeatures)
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
D = ones(1,size(xTrain,2))/size(xTrain,2);

classifiers = zeros(nbrWeakClassifiers);
min_classifier = zeros(123);

A      = zeros(1,nbrWeakClassifiers);
T      = zeros(1,nbrWeakClassifiers);
P      = zeros(1,nbrWeakClassifiers);
Haar   = zeros(1,nbrWeakClassifiers);
Errors = zeros(1,nbrWeakClassifiers);

for t = 1:nbrWeakClassifiers
    min_error = Inf;
    %haar_feature = 1 + round(rand()*(nbrHaarFeatures - 1));
    % Vi måste välja en haarfeature för varje svag klassifierare.
    % Vi måste på något sätt välja en haarfeature. Ett sätt är att slumpa
    % fram en. Ett annat är att låta vilken harfeature vi använder bero på
    % indexet på klassifieraren. Då får inte indexet på haarfeature-vectorn
    % vara för stor. Därför kan vi använda mod funktionen.
    haar_feature = 1 + mod(t,nbrHaarFeatures);
    Haar(t) = haar_feature;
    for threshold = xTrain(haar_feature,:)
        for polarity = [-1,1]
            C = WeakClassifier(threshold,polarity,xTrain(haar_feature,:));
            error = WeakClassifierError(C,D,yTrain);
            if error < min_error
                min_error = error;
                A(t) = 1/2*log((1-min_error)/min_error);
                T(t) = threshold;
                P(t) = polarity;
                Errors(t) = error;
            end
        end
    end
    D = D.*exp(-A(t)*yTrain.*WeakClassifier(T(t),P(t),xTrain(Haar(t),:)));
    D = D/sum(D);
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

C_train = StrongClassifier(A,T,P,xTrain,Haar);
train_error = StrongClassifierError(C_train,yTrain);
train_acc = 1 - train_error

C_test = StrongClassifier(A,T,P,xTest,Haar);
test_error = StrongClassifierError(C_test,yTest);
test_acc = 1 - test_error

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

Ytrain = zeros(1,nbrWeakClassifiers);
for N = 1:nbrWeakClassifiers
   Ytrain(N) = StrongClassifierError(StrongClassifier(A(1:N),T(1:N),P(1:N),xTrain,Haar(1:N)),yTrain);
end
Ytest = zeros(1,nbrWeakClassifiers);
for N = 1:nbrWeakClassifiers
   Ytest(N) = StrongClassifierError(StrongClassifier(A(1:N),T(1:N),P(1:N),xTest,Haar(1:N)),yTest);
end

figure(4);
hold on;
title('Error with respect to the number of weakclassifiers')
plot(1:nbrWeakClassifiers,Ytrain, 'r')
plot(1:nbrWeakClassifiers,Ytest, 'b')
xlabel('# Weakclassifiers')
ylabel('error')
legend('Training data','Test data')
str = "# faces = " + int2str(size(faces,3)) + " n # non-faces = " + int2str(size(nonfaces,3));
text(100,100,str)
hold off;

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.


figure(5);
colormap gray;
misclassified = find(C_test ~= yTest);
for k = 1:min(25,length(misclassified))
    subplot(5,5,k),imagesc(testImages(:,:,misclassified(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

[~,index] = mink(Errors, 25);
HaarsToPlot = Haar(index);

figure(6);
colormap gray;
for k = 1:min(25,nbrHaarFeatures)
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,HaarsToPlot(k)),[-1 2]);
    axis image;
    axis off;
end