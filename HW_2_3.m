s='y';
while s=='y'
    n=2; %number of feature dimensions;
    N=400; %number of iid samples
    mu(:,1)=input('please input the mean for class_1(mu(:,1))=');%[0,0]
    mu(:,2)=input('please input the mean for class_2(mu(:,2))=');%[3,3] 
    % mean for each class,[x-asix mean, y-axis mean
    Sigma(:,:,1)=input('please input the variance for class_1(Sigma(:,:,1))=');%eye(2)
    Sigma(:,:,2)=input('please input the variance for class_2(Sigma(:,:,2))=');%eye(2)
    %A 3-D array, for example, uses three subscripts. 
    %The first two are just like a matrix, but the third dimension represents pages or sheets of elements.
    p=input('please input the prior pobability for both cases[class_1,class_2]:');%[0.5,0.5]
    %class prioirs for labels 0 and 1 respectively
    label=rand(1,N)>=p(1); %obtain the bool value so that to choose each sample within which class
    %Uniformly distributed pseudorandom numbers.R = rand(N) returns an N-by-N matrix containing pseudorandom values drawn 
    %from the standard uniform distribution on the open interval(0,1).  
    Nc=[length(find(label==0)),length(find(label==1))];% number of samples from each class
    x=zeros(n,N);%save up space
    %draw samples from each class pdf
    for l=0:1
        x(:,label==l)=mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
    %R = mvnrnd(MU,SIGMA,N) returns a N-by-D matrix R of random vector chosen
    %from the multivariate normal distribution with 1-by-D mean vector MU, and
    %D-by-D covariance matrix SIGMA. In this case is (Nc-by-2)'=(2-by-Nc)
    %x(:,label==1)=randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1)
    end
    figure(1)
    plot(x(1,label==0),x(2,label==0),'o');
    hold on
    plot(x(1,label==1),x(2,label==1),'+');
    hold off
    axis equal
    legend('Class 0','Class 1')
    title('Data and their true labels')
    xlabel('x_1')
    ylabel('x_2')

    %loss value after LDA(for MAP, choose 0-1 loss)
    lambda=[0 1;1 0];
    %generate the function of threshold gamma based on risk function
    gamma=(lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*p(1)/p(2);
    %take log on both side, which turns division to minus,p(x|w2)-p(x|w1):
    discriminationScore=log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
    decision=(discriminationScore>=log(gamma));
    %probability  of true negative:
    ind00=find(decision==0&label==0);p00=length(ind00)/Nc(1);
    %probability of false positive:
    ind10=find(decision==1&label==0);p10=length(ind10)/Nc(1);%number and probability of errors that decision==1 and label==0  
    %probability of false negative:
    ind01=find(decision==0&label==1);p01=length(ind01)/Nc(2);%number and probability of errors that decision==0 and label==1 
    %probability of true positive:
    ind11=find(decision==1&label==1);p11=length(ind11)/Nc(2);
    p_error=[p10 p01]*Nc'/N;%p=(p10*Nc(1)+p01*Nc(2))/N;
    fprintf('the total number of errors are %d,\n',length(ind01)+length(ind10))
    fprintf('the probability of errors is %f,\n',p_error)
    figure(2)
    %class 0 circle,class 1 +,correct green,incorrect red
    plot(x(1,ind00),x(2,ind00),'og');%samples(x1,x2) that belong to label 0 , decision=0;
    hold on
    plot(x(1,ind10),x(2,ind10),'or')%samples(x1,x2) that belong to label 0, decision=1;
    hold on
    plot(x(1,ind01),x(2,ind01),'+r')%samples(x1,x2) that belong to label 1, desicion =0;
    hold on
    plot(x(1,ind11),x(2,ind11),'+g')%samples(x1,x2) that belong to label 1, desicion=1;
    legend('Label=0 & Decision=0','Label=0 & Decision=1','Label=1 & Decision=0','Label=1 & Decision=1')
    xlabel('x_1')
    ylabel('x_2')
    title('data along with their inferred(decision) labels')
    axis equal
    %Draw the decision boundary
    %horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
    %verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
    %[h,v] = meshgrid(horizontalGrid,verticalGrid);
    %discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
    %minDSGV = min(discriminantScoreGridValues);
    %maxDSGV = max(discriminantScoreGridValues);
    %discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
    %figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
    %including the contour at level 0 which is the decision boundary
    %legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
    %title('Data and their classifier decisions versus true labels'),
    %xlabel('x_1'), ylabel('x_2'), 


    % Using fisherLDA to seperate data and generate the mean and variance
    Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
    Sw = Sigma(:,:,1) + Sigma(:,:,2);
    [V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies lambda(eigenvalue)*Sw*w = Sb*w; ie w is a generalized eigenvector of (Sw,Sb)
    % equivalently alpha w  = inv(Sw)*Sb*w
    % D is the a diagonal matrix of generalized eigenvalues;
    [~,ind] = sort(diag(D),'descend');%ignore the result, only care about the index;
    %the optimal w is the eigenvector of inv(Sw)*Sb that corresponds to the
    %largest eigenvalue of this matrix
    wLDA = V(:,ind(1)); % Fisher LDA projection vector
    yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
    wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
    yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
    figure(3), clf,
    plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
    plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
    legend('Class 0','Class 1'), 
    title('LDA projection of data and their true labels'),
    xlabel('x_1'), ylabel('x_2'), 
    mu_0=mean(yLDA(label==0));%mean of samples in yLDA that belong to label==0 
    mu_1=mean(yLDA(label==1));%mean of samples in yLDA that belong to label==1
    Var_0= var(yLDA(label==0));
    Var_1=var(yLDA(label==1));
    discriminationScore_LDA=log(evalGaussian(x,mu_1,Var_1))-log(evalGaussian(x,mu_0,Var_0));
    decision_LDA=(discriminationScore_LDA>=log(gamma));
    %probability  of true negative:
    ind_LDA00=find(decision_LDA==0&label==0);p_LDA00=length(ind_LDA00)/Nc(1);
    %probability of false positive:
    ind_LDA10=find(decision_LDA==1&label==0);p_LDA10=length(ind_LDA10)/Nc(1);%number and probability of errors that decision==1 and label==0  
    %probability of false negative:
    ind_LDA01=find(decision_LDA==0&label==1);p_LDA01=length(ind_LDA01)/Nc(2);%number and probability of errors that decision==0 and label==1 
    %probability of true positive:
    ind_LDA11=find(decision_LDA==1&label==1);p_LDA11=length(ind_LDA11)/Nc(2);
    p_error_LDA=[p_LDA10 p_LDA01]*Nc'/N;%p=(p10*Nc(1)+p01*Nc(2))/N;
    fprintf('the total number of errors are %d,\n',length(ind_LDA01)+length(ind_LDA10))
    fprintf('the probability of errors is %f,\n',p_error_LDA)
    s=input('do you want to continue?:(y/n)','s');
    end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^(-n/2) * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
