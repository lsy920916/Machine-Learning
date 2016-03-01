function [Model] = train(X,Y)
X_hog = [];
N = size(X,1);
new_Y = zeros(N,10);

% Feature extraction and labels vector reallocation
for i=1:N
    hog = HOG(X(i,:));
    X_hog = [X_hog;hog'];
    if(Y(i) == 0)
        new_Y(i,10) = 1;
    else
        new_Y(i,Y(i))=1;
    end
end
X_hog = [X_hog ones(N,1)];
F = size(X_hog,2);

% parameter initialization
H = 100;
O = 10;
s = 0.002;
B1 = (6/(F+H))^0.5;
B2 = (6/(H+O))^0.5;
wh = -B1+B1*rand(F,H);
wo = -B2+B2*rand(H,O);

% training process
for iter = 1:50
    for i = 1:N
        [dw_h,dw_o] = BPP(X_hog(i,:),new_Y(i,:),wh,wo,H,O,s);
        wh = wh+dw_h;
        wo = wo+dw_o;
    end
end

Model.wh = wh;
Model.wo = wo;
save('Model.mat','Model');
end

% back propagation funxtion
function [dw_h,dw_o]=BPP(X,Y,wh,wo,H,O,s)
v_h = zeros(1,H);
v_o = zeros(1,O);
dw_h = zeros(size(wh));
dw_o = zeros(size(wo));
mu_h = zeros(1,H);
N = length(X);

for i = 1:H
    v_h(i) = sigmoid(X,wh(:,i));
end

for i = 1:O
    v_o(i) = sigmoid(v_h,wo(:,i));
end

mu_o = v_o.*(ones(1,O)-v_o).*(Y-v_o);

for i=1:H
    mu_h(i) = v_h(i)*(1-v_h(i))*wo(i,:)*mu_o';
end

for i = 1:H
    for j = 1:O
        dw_o(i,j) = s*mu_o(j)*v_h(i);
    end
end

for i = 1:N
    for j = 1:H
        dw_h(i,j) = s*mu_h(j)*X(i);
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