n=2; %number of feature dimensions
N=400; %number of iid samples
mu(:,1)=[0,0]; mu(:,2)=[3;3];% mean for each class,[x-asix mean, y-axis mean]
Sigma(:,:,1)=eye(2); Sigma(:,:,2)=eye(2);%A 3-D array, for example, uses three subscripts. 
%The first two are just like a matrix, but the third dimension represents pages or sheets of elements.
p=[0.5,0.5]% class prioirs for labels 0 and 1 respectively
label=rand(1,N)>=p(1) %obtain the bool value so that to choose wchich sample is in which class
%Uniformly distributed pseudorandom numbers.R = rand(N) returns an N-by-N matrix containing pseudorandom values drawn 
%from the standard uniform distribution on the open interval(0,1).  
Nc=[length(find(label==0)),length(find(label==1))]% number of samples from each class
x=zeros(n,N);%save up space

%draw samples from each class pdf
for l=0:1
    x(:,label==l)=mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))'
%R = mvnrnd(MU,SIGMA,N) returns a N-by-D matrix R of random vector chosen
%from the multivariate normal distribution with 1-by-D mean vector MU, and
%D-by-D covariance matrix SIGMA. In this case is (Nc-by-2)'=(2-by-Nc)
%x(:,label==1)=randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1)
end

plot(x(1,label==0),x(2,label==0),'o');
hold on
plot(x(1,label==1),x(2,label==1),'+');
hold off
axis equal
legend('Class 0','Class 1')
title('Data and their true labels')
xlabel('x_1')
ylabel('x_2')


    

 
