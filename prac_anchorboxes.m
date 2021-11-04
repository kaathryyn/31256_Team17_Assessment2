
This code loads a table containing filenames and corresponding bounding boxes.
rng(0)
load petGroundTruth.mat
imds = imageDatastore(petGT.imageFilename);
bxds = boxLabelDatastore(petGT(:,2:end));
data = combine(imds,bxds);
scaledData = transform(data,@scaleGT);

Task 1

anchorBoxes = estimateAnchorBoxes(scaledData, 5)

Task 2

net = resnet18
numClasses = 5
imageSize = [224 224 3]

Task 3

lgraph=yolov2Layers(imageSize, numClasses, anchorBoxes, net, 'res5b_relu', 'ReorgLayerSource','res3a_relu')

Further Practice



scaleGT resizes images to targetSize. It also uses the same scale to resize the corresponding bounding boxes.
function data = scaleGT(data)  
    targetSize = [224 224];
    % data{1} is the image
    scale = targetSize./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize);
    % data{2} is the bounding box
    data{2} = bboxresize(data{2},scale);
end
