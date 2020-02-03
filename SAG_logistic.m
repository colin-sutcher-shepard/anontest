function [w,Out] = SAG_logistic(X,y,mu,opts)
% min_w 1/N*sum_log(1+exp(-y.*(X'*w)))  + mu/2*norm(w)^2

% N: number of samples
% p: number of features

[N,p] = size(X);

maxepoch = 100;
tol = 1e-7;


if isfield(opts,'maxepoch')        maxepoch = opts.maxepoch;             end
if isfield(opts, 'w0')             wOld = opts.w0;                         end
if isfield(opts, tol)              tol = opts.tol;                       end

Y = zeros(p,N);
d = zeros(size(wOld));
Ls = 1/4;
alpha = 1/16*1/Ls;
for i = 1:N
    Y(:,i) = -1*y(i)/(1+exp(y(i).*X(i,:)*wOld)).*X(i,:)';
    d = d + Y(:,i);
end
wNew = wOld - alpha/(N).*d - alpha* mu.*wOld;

hist_obj = [];




for epoch = 1:maxepoch
    for k = 2:N
        %select index uniformly at random
        ik = randi(N);
        oldGrad = Y(:,ik);
        newGrad = -1*y(ik)/(1+exp(y(ik)*(X(ik,:)*wNew))).*X(ik,:)';
        Y(:,ik) = newGrad;
        wOld = wNew;
        d = d - oldGrad+newGrad;
        wNew = wOld - alpha/N.*d - alpha*mu.*wOld;
        
    end
    
    % compute objective value after every epoch
    objc = 1/N*sum(log(1+exp(-1*y.*(X*wNew))))+mu/2*norm(wNew)^2;
    hist_obj= [hist_obj, objc];
end

% save the history objective to Out
Out.obj = hist_obj;
w = wNew;
end
