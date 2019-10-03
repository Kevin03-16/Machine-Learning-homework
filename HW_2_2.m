n=2; %number of feature dimensions;
N=400; %number of iid samples
mu(:,1)=[0,0];
mu(:,2)=[3;3]% mean for each class,[x-asix mean, y-axis mean
Sigma(:,:,1)=eye(2)
Sigma(:,:,2)=eye(2)%A 3-D array, for example, uses three subscripts. 
%The first two are just like a matrix, but the third dimension represents pages or sheets of elements.
p=[0.5,0.5]% class prioirs for labels 0 and 1 respectively
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

%loss value(for MAP, choose 0-1 loss)
lambda=[0 1;1 0];
%generate the function of threshold gamma based on risk function
gamma=(lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*p(1)/p(2);
%take log on both side, which turns division to minus,p(x|w2)-p(x|w1):
discriminationScore=log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
decision=(discriminationScore>=log(gamma));
%probability  of true negative:
ind00=find(decision==0&label==0);p00=length(ind00)/Nc(1);
%probability of false positive:
ind10=find(decision==1&label==0);
length(ind10)
p10=length(ind10)/Nc(1)%number and probability of errors that decision==1 and label==0  
%probability of false negative:
ind01=find(decision==0&label==1);
length(ind01)
p01=length(ind01)/Nc(2)%number and probability of errors that decision==0 and label==1 
%probability of true positive:
ind11=find(decision==1&label==1);p11=length(ind11)/Nc(2);
%p(error)=[p10 p01]*Nc'/N；p=(p10*Nc(1)+p01*Nc(2))/N;

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
ylabel('y_1')
title('data along with their inferred(decision) labels')
hold off
axis equal

    

 
