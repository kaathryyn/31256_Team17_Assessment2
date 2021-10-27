load petGroundTruth.mat
head(petGroundTruth)
im = imread(petGroundTruth.imageFilename{2})
bbox1=petGroundTruth.Lucy{2}
annotationLabel1='Lucy'
labeledim = insertObjectAnnotation(im,'rectangle', bbox1, annotationLabel1)
imshow(labeledim)

load petGroundTruth.mat
petGT

Task 1
imds=imageDatastore(petGT.imageFilename)


Task 2

bxds=boxLabelDatastore(petGT(:,2:end))

Task 3
data = combine(imds, bxds)


Task 4
scaledData=transform(data,@scaleGT)


Task 5

newGT=preview(scaledData)

Task 6
im = insertObjectAnnotation(newGT{1}, 'rectangle', newGT{2}, newGT{3})
imshow(im)

