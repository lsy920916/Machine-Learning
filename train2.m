function [Model] = train(X,Y)
X_hog = [];
N = size(X,1);

% feature extraction
for i=1:N
    hog = HOG(X(i,:));
    X_hog = [X_hog;hog'];
end

% parameter initialization
new_Y = -ones(N,10);
alpha = zeros(N,10);
b = zeros(10,1);

for y = 0:9  
    % label vector reallocation
    index = find(Y==y);
    for i=1:size(index)
        new_Y(index(i),y+1)=1;
    end

    H = zeros(N,N);
    for i = 1:N
        for j=1:N
            H(i,j) = new_Y(i,y+1)*new_Y(j,y+1)*X_hog(i,:)*X_hog(j,:)';
        end
    end
    
    % optimization process
    f = -ones(N,1);
    Aeq = new_Y(:,y+1)';
    beq = 0;
    lb = zeros(N,1);
    ub = 10*ones(N,1);
    alpha(:,y+1) = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    
    sv = find(alpha(:,y+1) > 1e-5);
    len = length(sv); 
    
    % calculate b
    for n=1:len
        tmp = 0;
        for m=1:len
            tmp = tmp+alpha(sv(m),y+1)*new_Y(sv(m),y+1)*...
                X_hog(sv(n),:)*X_hog(sv(m),:)';
        end
        b(y+1) = b(y+1)+1/len*(new_Y(sv(n),y+1)-tmp);
    end
end

Model.alpha = alpha;
Model.b = b;
Model.Y = new_Y;
Model.X = X_hog;
save('Model.mat','Model')
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