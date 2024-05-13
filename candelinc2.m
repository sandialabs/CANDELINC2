function [iSr,jXr,kTr,qRr,RMSE,i] = ...
    candelinc2(iDjk,r,NN,thresh,maxit,fQP,iSr,jXr,kTr,qRr)
% Three mode decomposition PARAFAC2 using CANDELINC data compression approach.
%
% I/O:  [iSr,jXr,kTr,qRr,RMSEE,iter] = ...
%               candelinc2(iDjk,r,NN,thresh,maxit,fQP,iSr,jXr,kTr,qRr)
%       [iSr,jXr,kTr] = candelinc2(iDjk,r,NN,thresh,maxit)
%
% INPUTS:
% iDjk:      Data entered as an IxKxJ array, with j-outer loop, k-inner 
%            loop, if the data can be entered as a matrix. Otherwise data 
%            must be a 3D cell array where each cell is IxK.
% r:         Rank (r) of the decomposition of the data.
% Optional Inputs:
% NN:        Number of nonnegative iterations for each mode (iSr,jXr,kTr),
%            if scalar entered, it applies to all modes
% thresh:    Termination RMSE difference between consecutive iterations.
%            Enter 'auto' (default) to use automatic termination criterion.
% maxiter:   Termination maximum number of iterations.
% fQP:       Set to true if data is very big and memory errors occur.
%            This option splits up the nonnegative least squares 
%            calulations when the second and\or third modes are very big.
% iSr,jXr,kTr: Initial estimates for iSr, jXr and kTr modes, where
%            r is the rank of the trilinear decomposition.
% qRr:       Initial estimates for the rotation matrix of the non-rigid 
%            axis mode, where kTr{k} = kPp{k}*pQq{k}*qRr.
%
% OUTPUTS:
% iSr,jXr,kTr: Model estimates for iSr, jXr and kTr modes, where
%            r is the rank of the trilinear decomposition.
% qRr:       Model estimates for the rotation matrix of the non-rigid 
%            axis mode, where kTr{k} = kPp{k}*pQq{k}*qRr.
% RMSE:      Root mean square error of model fit with data for each
%            iteration.
% Iter:      Number of iterations of the algorithm performed.
%
% Dependencies: FCNNLS.M
%
% Copyright 2019 National Technology & Engineering Solutions of Sandia, 
% LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the 
% U.S. Government retains certain rights in this software.

% M.H. Van Benthem, Sandia National Laboratories, 1/22/2016
% Reference: GETTING TO THE CORE OF PARAFAC2, A NONNEGATIVE APPROACH–
% Mark H. Van Benthem, J. Chemom, TBD
% Revised: 2/22/2017, 5/8/17, 5/9/17, 8/9/19

% create a core matrix for the current design
narginchk(2,10)
if iscell(iDjk)
    sD = length(iDjk);    % the number of slabs in D
    cD = zeros(1,sD);  % the number of columns in each slab of D
    for k = 1:sD
        if k==1
            rD = size(iDjk{k},1); % the number of rows in D
        elseif size(iDjk{k},1)~=rD
            error('Each slab in D must have the same number of rows')
        end
        cD(k) = size(iDjk{k},2);
    end
elseif ndims(iDjk)==3
    [rD,cD,sD] = size(iDjk);
    tempD = cell(1,sD);
    for k = 1:sD
        tempD{k} = iDjk(:,:,k);
    end
    iDjk = tempD;
    clear('tempD')
    cD = repmat(cD,1,sD);
else
    error('The data in D must be a cell or three dimensional array')
end
if nargin < 3
    NN = zeros(1,3);
elseif length(NN) == 1
    NN = repmat(NN,1,3);
elseif length(NN) ~= 3
    error('Specify only one or three nonneg entries')
else % reorder NN to match order of PARAFAC solution
    NN = NN([1 3 2]);
end
if nargin < 4 || strcmpi(thresh,'auto') || isempty(thresh)
    thresh = 1e-6;
end
if nargin < 5
    maxit = 10000;
end
if nargin < 6
    fQP = false;
end
if nargin<7||isempty(iSr), rs = rD; cs = r; else, [rs,cs] = size(iSr); end
if nargin<8||isempty(jXr), rx = sD; cx = r; else, [rx,cx] = size(jXr); end
if nargin<9||isempty(kTr)
    rt = cD;
    ct = repmat(r,1,sD);
elseif ~isempty(kTr) && (~iscell(kTr) || length(kTr)~=sD)
    error('kTr must be a cell array with size of third dimension of D')
else
    rt = zeros(1,sD);
    ct = zeros(1,sD);
    for k = 1:sD
        [rt(k),ct(k)] = size(kTr{k});
    end
end
if nargin<10||isempty(qRr), rr = r;  cr = r; else, [rr,cr] = size(qRr); end
if rs~=rD || rr~=r || rx~=sD || any(rt~=cD)
    error('Array size does not match iSr, kTr, or jXr dimensions')
elseif cs~=r || cr~=r || cx~=r || any(ct~=r)
    error('iSr, kTr, or jXr dimensions do not match desired rank')
end

SSE = zeros(1,maxit);
p = [min(rD,r) min(sD,r) min(rD,r)];
odr = 1:3;
Dmag = zeros(rD,rD);
for k = 1:sD
    Dmag = Dmag + iDjk{k}*iDjk{k}';
end
[iOm,~] = svds(Dmag,p(odr(1)));
Dmag = [trace(Dmag) 0 0];
kPp = cell(1,sD);
pQq = cell(1,sD);
qYp = cell(1,sD);
if exist('iSr','var')&&~isempty(iSr)
    mHr = iOm'*iSr;
    iSr = iSr.';
elseif NN(1)>0
    iSr = rand(r,rD);
    mHr = (iSr*iOm)';
else
    mHr = eye(p(odr(1)),r);
    mHr(:,all(mHr==0,1)) = rand(p(odr(1)),sum(all(mHr==0,1)))./p(odr(1));
end
if exist('jXr','var')&&~isempty(jXr)
    jXr = jXr.';
else
    jXr = ones(r,sD) + randn(r,sD)/10;
end
if exist('kTr','var')&&~isempty(kTr)
    inT = true;
else
    kTr = cell(1,sD);
    inT = false;
end
if ~exist('qRr','var')||isempty(qRr)
    qRr = eye(p(odr(3)),r);
    qRr(:,all(qRr==0,1)) = rand(p(odr(3)),sum(all(qRr==0,1)))./p(odr(3));
end
mGjq = zeros(p(odr(1)),p(odr(3)),sD);
if ~fQP && NN(3)>0
    rjk = [0 cumsum(cD)];
    QP = zeros(p(odr(3)),rjk(end));
end
for k = 1:sD
    [u,s,v] = svd(iOm'*iDjk{k},'econ');
    kPp{k} = v;
    qYp{k} = u*s;
    if inT
        pQq{k} = kPp{k}'*(kTr{k}/qRr);
    else
        [u,~,v] = svd(qRr*diag(jXr(:,k))*mHr'*qYp{k},'econ');
        pQq{k} = v*u';
    end
    mGjq(:,:,k) = qYp{k}*pQq{k};
end
[mWnq,s,jCn] = svd(reshape(mGjq,p(odr(1))*p(odr(3)),sD),'econ');
mWnq = mWnq(:,1:p(odr(2)))*s(1:p(odr(2)),1:p(odr(2)));
jCn = jCn(:,1:p(odr(2)));
mWnq = reshape(mWnq,p(odr(1)),p(odr(3))*p(odr(2))).';
Vo = jCn;
hats = {mHr (jXr*jCn)' qRr};
xps = {hats{1}'*hats{1} hats{2}'*hats{2} hats{3}'*hats{3}};
% Create the output string to monitor progress
nitstr = floor(log10(maxit))+1;
outstr = sprintf('%%%dd, SSE: %%10.4e PVE:%%6.2f\\n',nitstr);
backstr = repmat('\b',1,nitstr+29);
fprintf(['Iteration: ',outstr],0,0,0);
intit = 5;
twofd = tril(true(r,r),-1);
UCC = cell(1,3);
chk2fd = 0;
for i = 1:maxit
    for j = 1:3*intit
        q = p(odr(3))*p(odr(2));
        ks = zeros(q,p(odr(1)));
        for k = 1:r
            ks(:,k) = reshape(hats{odr(3)}(:,k)*hats{odr(2)}(:,k).',q,1);
        end
        if NN(odr(1))<i % uncontrained least-squares
            hats{odr(1)} = ((xps{odr(2)}.*xps{odr(3)})\ks'*mWnq).';
        elseif odr(1)==1 % FCNNLS for iSr mode
            iSr = fcnnls((xps{odr(2)}.*xps{odr(3)}),ks'*mWnq*iOm',iSr,...
                iSr>0,zeros(size(iSr)),0);
            hats{odr(1)} = (iSr*iOm)';
        elseif odr(1)==2 % FCNNLS for jXr mode
            jXr = fcnnls((xps{odr(2)}.*xps{odr(3)}),ks'*mWnq*jCn',jXr,...
                jXr>0,zeros(size(jXr)),0);
            hats{odr(1)} = (jXr*jCn)';
        else % FC_NNLS for qRr-pQq mode
            ohat = hats{odr(1)};
            hats{odr(1)} = zeros(p(odr(1)),r);
            knn = ks'*mWnq;
            if fQP
                for k = 1:sD % solve FCNNLS for each
                    QP = (kPp{k}*pQq{k}).';
                    fkold = ohat'*QP; % previous solution
                    fk = fcnnls(xps{odr(2)}.*xps{odr(3)},knn*QP, ...
                        fkold, fkold>0, zeros(size(fkold)), 0);
                    hats{odr(1)} = hats{odr(1)}+(fk*QP.').';
                end
            else
                for k = 1:sD
                    QP(:,rjk(k)+1:rjk(k+1)) = (kPp{k}*pQq{k}).';
                end
                fkold = ohat'*QP; % previous solution
                fk = fcnnls(xps{odr(2)}.*xps{odr(3)},knn*QP, ...
                    fkold, fkold>0, zeros(size(fkold)), 0);
                hats{odr(1)} = QP*fk.';
            end
        end
        if any(all(hats{odr(1)}==0,1))
            error('Factor rank too low, consider another starting point.')
        end
        if odr(1)~=2
            hats{odr(1)} = hats{odr(1)}*...
                diag(1./sqrt(sum(abs(hats{odr(1)}).^2)));
        end
        xps{odr(1)} = hats{odr(1)}'*hats{odr(1)};
        odr = circshift(odr,-1,2);
        mWnq = reshape(mWnq.',p(odr(2))*p(odr(3)),p(odr(1)));
    end
    % check for 2-factor degeneracy
    for j = 1:3
        UCC{j} = abs(xps{j}./sqrt(diag(xps{j})*diag(xps{j})'));
        UCC{j} = 1-UCC{j}(twofd)<1e-9;
    end
    if any(UCC{1}&UCC{2})||any(UCC{1}&UCC{3})||any(UCC{2}&UCC{3})
        chk2fd = chk2fd + 1;
        if chk2fd==3
            error('Detected 2FD, try new initial values.')
        end
        warning('Potential 2-factor degeneracy')
        fprintf(['\nIteration: ',outstr],0,0,0);
    end
    mWnq_hat = reshape(ks*hats{odr(3)}',p(odr(2)),p(odr(1))*p(odr(3)))';
    for k = 1:sD
        [u,~,v] = svd(reshape(mWnq_hat*jCn(k,:)',p(odr(1)),...
            p(odr(3)))'*qYp{k},'econ');
        pQq{k} = v*u';
        mGjq(:,:,k) = qYp{k}*pQq{k};
    end
    [mWnq,s,jCn] = svd(reshape(mGjq,p(odr(1))*p(odr(3)),sD),'econ');
    mWnq = mWnq(:,1:p(odr(2)))*s(1:p(odr(2)),1:p(odr(2)));
    jCn = jCn(:,1:p(odr(2)));
    sm = diag(sign(sum(jCn.*Vo)));
    jCn = jCn*sm;
    Vo = jCn;
    mWnq = reshape(mWnq*sm,p(odr(1)),p(odr(3))*p(odr(2)));
    mWnq_hat = reshape(mWnq_hat,p(odr(1)),p(odr(3))*p(odr(2)));
    % Compute SSE
    Dmag(2) = -2*sum(sum(mWnq.*mWnq_hat));
    Dmag(3) = sum(sum(mWnq_hat.*mWnq_hat));
    SSE(i) = sum(Dmag);
    mWnq = mWnq.';
    fprintf(backstr);fprintf(outstr,i,SSE(i),100*(1-SSE(i)/Dmag(1)));
    if i>5 && abs(SSE(i-1)-SSE(i))<=SSE(i-1)*thresh % do at least 5 iters
        break
    end
end
iSr = iOm*hats{1};
jXr = jCn*hats{2};
qRr = hats{3};
RMSE = sqrt(SSE(1:i)/sum(rD*cD));
for k = 1:sD
    kTr{k} = kPp{k}*pQq{k}*qRr;
end
end

