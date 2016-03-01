function [y]=classify(Model,X)
X_hog = [];
N = size(X,1);
for i=1:N
    hog = HOG(X(i,:));
    X_hog = [X_hog;hog'];
end

X = Model.X;
Y = Model.Y;
b = Model.b;
alpha = Model.alpha;
y = zeros(N,1);
sv = cell(10,1);
X_t = cell(10,1);
values = zeros(10,1);

for i = 1:10
    sv{i} = find(alpha(:,i) > 1e-5);
    Z = bsxfun(@times,alpha(:,i).*Y(:,i),X);
    X_t{i} = Z(sv{i},:);
end

for i=1:N
    for k = 1:10
        values(k) = sum(X_t{k}*X_hog(i,:)')+b(k);
    end
    max_index = find(values == max(values));
    y(i) = max_index-1;
end

end

function [feat] = HOG(X)
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