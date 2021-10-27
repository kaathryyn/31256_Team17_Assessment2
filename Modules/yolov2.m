
layers = [
    imageInputLayer([256 256 3],"Name","input","Normalization","rescale-zero-one")
    convolution2dLayer([3 3],32,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm1")
    leakyReluLayer(0.1,"Name","leaky1")
    maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm2")
    leakyReluLayer(0.1,"Name","leaky2")
    maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","batchnorm3")
    leakyReluLayer(0.1,"Name","leaky3")
    convolution2dLayer([1 1],64,"Name","conv4","Padding","same")
    batchNormalizationLayer("Name","batchnorm4")
    leakyReluLayer(0.1,"Name","leaky4")
    convolution2dLayer([3 3],128,"Name","conv5","Padding","same")
    batchNormalizationLayer("Name","batchnorm5")
    leakyReluLayer(0.1,"Name","leaky5")
    maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv6","Padding","same")
    batchNormalizationLayer("Name","batchnorm6")
    leakyReluLayer(0.1,"Name","leaky6")
    convolution2dLayer([1 1],128,"Name","conv7","Padding","same")
    batchNormalizationLayer("Name","batchnorm7")
    leakyReluLayer(0.1,"Name","leaky7")
    convolution2dLayer([3 3],256,"Name","conv8","Padding","same")
    batchNormalizationLayer("Name","batchnorm8")
    leakyReluLayer(0.1,"Name","leaky8")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv9","Padding","same")
    batchNormalizationLayer("Name","batchnorm9")
    leakyReluLayer(0.1,"Name","leaky9")
    convolution2dLayer([1 1],256,"Name","conv10","Padding","same")
    batchNormalizationLayer("Name","batchnorm10")
    leakyReluLayer(0.1,"Name","leaky10")
    convolution2dLayer([3 3],512,"Name","conv11","Padding","same")
    batchNormalizationLayer("Name","batchnorm11")
    leakyReluLayer(0.1,"Name","leaky11")
    convolution2dLayer([1 1],256,"Name","conv12","Padding","same")
    batchNormalizationLayer("Name","batchnorm12")
    leakyReluLayer(0.1,"Name","leaky12")
    convolution2dLayer([3 3],512,"Name","conv13","Padding","same")
    batchNormalizationLayer("Name","batchnorm13")
    leakyReluLayer(0.1,"Name","leaky13")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    convolution2dLayer([3 3],1024,"Name","conv14","Padding","same")
    batchNormalizationLayer("Name","batchnorm14")
    leakyReluLayer(0.1,"Name","leaky14")
    convolution2dLayer([1 1],512,"Name","conv15","Padding","same")
    batchNormalizationLayer("Name","batchnorm15")
    leakyReluLayer(0.1,"Name","leaky15")
    convolution2dLayer([3 3],1024,"Name","conv16","Padding","same")
    batchNormalizationLayer("Name","batchnorm16")
    leakyReluLayer(0.1,"Name","leaky16")
    convolution2dLayer([1 1],512,"Name","conv17","Padding","same")
    batchNormalizationLayer("Name","batchnorm17")
    leakyReluLayer(0.1,"Name","leaky17")
    convolution2dLayer([3 3],1024,"Name","conv18","Padding","same")
    batchNormalizationLayer("Name","batchnorm18")
    leakyReluLayer(0.1,"Name","leaky18")
    convolution2dLayer([1 1],9,"Name","conv19","Padding","same")
    globalAveragePooling2dLayer("Name","avg1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output")];
