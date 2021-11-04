clc
clear all
close all

doTraining = true;

if ~doTraining
    preTrainedDetector = downloadPretrainedYOLOv3Detector();    
end

data = load("gTruth.mat");
covidDataset = data.gTruth.LabelData;
covidDataset.imageFilename = data.gTruth.DataSource.Source;
head(covidDataset)
covidDataset.imageFilename = fullfile(covidDataset.imageFilename);

% split the data into training and test set
rng(0);
shuffledIndices = randperm(height(covidDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = covidDataset(shuffledIndices(1:idx), :);
testDataTbl = covidDataset(shuffledIndices(idx+1:end), :);

% Creating datastores for image and label data
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);

bldsTrain = boxLabelDatastore(trainingDataTbl(:, 1:5));
bldsTest = boxLabelDatastore(testDataTbl(:, 1:5));

trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

%augment training data
augmentedTrainingData = transform(trainingData, @augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)

networkInputSize = [227 227 3];

rng(0)
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 5;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:5,:)
    };
%pre trained network
baseNetwork = squeezenet;
classNames = trainingDataTbl.Properties.VariableNames(1:5);

yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'});

% ERROR - preprocessing
preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(yolov3Detector, data));
data = read(preprocessedTrainingData);
I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
reset(preprocessedTrainingData);

numEpochs = 80;
miniBatchSize = 8;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

%train model
if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);