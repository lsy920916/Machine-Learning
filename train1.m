function [Model] = train1(X,Y)
N = size(X,1);
feat = HOG(X(1,:))';
F = size(feat,2);
features = zeros(N,F);

% Generate features of HOG
% Since we use K-Nearest Neibor classification, train the model is just to 
% get feature matrix
for i=1:N
    features(i,:) = HOG(X(i,:))';
end

% Model consists of two parts(feature matrix, label vector)
Model.features = features;
Model.labels = Y;
save('Model.mat','Model');
end

function feat = HOG(X)
image_RGB = im2single(reshape(X, 32, 32, 3));
feature = extract_feature(image_RGB);
feat = feature(:);
end

function [feat] = extract_feature(image)
if ~isa(image, 'single'), image = single(image); end;
cellSize = 8;
hog = vl_hog(image, cellSize);
feat = hog(:);
end