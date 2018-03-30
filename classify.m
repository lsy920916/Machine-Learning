function [y] = classify(Model,X)
wh = Model.wh;
wo = Model.wo;
H = size(wh,2);
O = 10;
N = size(X,1);
y = zeros(N,1);

v_h = zeros(1,H);
v_o = zeros(1,O);

% Git test
for i = 1:N
    hog = HOG(X(i,:));
    X_hog = [hog',1];
    for j = 1:H
        v_h(j) = sigmoid(X_hog,wh(:,j));
    end
    
    for j = 1:O
        v_o(j) = sigmoid(v_h,wo(:,j));
    end
    
    index = find(v_o == max(v_o));
    
    if(index == 10)
        y(i) = 0;
    else
        y(i) = index(1);
    end
end


end

function [v] = sigmoid(X,w)
v = 1/(1+exp(-X*w));
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