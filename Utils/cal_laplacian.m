function L = cal_laplacian(W)
    D                                 = sum(W,2);
    D(D~=0)                           = sqrt(1./D(D~=0));
    D                                 = spdiags(D,0,speye(size(W,1)));
    W                                 = D*W*D;
    L                                 = speye(size(W,1))-W; % L = I-D^-1/2*W*D^-1/2
end