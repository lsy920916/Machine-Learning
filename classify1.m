function [y] = classify1(Model,X)
% load Model.mat to get model variables(feature matrix and label vector)
features = Model.features;
labels = Model.labels;

N = size(X,1);
cellsize = 8;

% y is the prediction set
y = zeros(N,1);

% k is the neiborhood size
k = 20;

% Extract features from test set and then classify it by KNN
for i = 1:N
    % feature extraction
    image_RGB = im2single(reshape(X(i,:), 32, 32, 3));
    hog = vl_hog(image_RGB,cellsize);
    hog = hog(:)';
    % classification
    y(i) = KNN(features,hog,labels,k);
end

end

% K-Nearest Neighbor classification
% Takes trained neighborhood points, test set, label vetor, and
% neighborhood size as input
function [label] = KNN(points,XTest,labels,k)
N = size(points,1);

% initilize distance vector of all neighborhood points and correspoding
% labels
rank = zeros(N,2);
rank(:,2) = labels;

% compute distances between test data and all neighborhood points 
for i = 1:N
    rank(i,1) = norm(points(i,:)-XTest);
end

% sort the rank matrix by accending distances (labels are also permutated)
[values,order] = sort(rank(:,1));
sortedRank = rank(order,:);

% return the most frequent label in k nearest neighbors
label = mode(sortedRank(1:k,2));  
end