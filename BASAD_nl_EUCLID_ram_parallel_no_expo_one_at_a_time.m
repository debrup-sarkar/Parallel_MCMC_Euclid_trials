clc;clear;close;

% warning('off', 'specific_warning_identifier')
warning('off', 'EMLRT:runTime:MATLABSourceModified')

%% Synthetic data


% Imported the FEM data generated using forward FEA algorithm

data = load('D:\ds\EUCLID-plasticity-main\EUCLID-plasticity-main\FEM_data\20240110T165055_DATA_FYS_1_HardMix_plate_elliptic_holes.mat');
data = data.data;
in = opt_input_HardMix();
mat = FYS_1_HardMix();
dof_free = setdiff(data.mesh.dof, [data.bc.dof_fix, data.bc.dof_displacement]);
reaction = data.results.reaction;
dof_reaction = data.bc.dof_reaction;
n_eval_step = length(1:in.n_eval:size(reaction,2));
n_residual = length(dof_free) + size(dof_reaction,1);

% Nonlinear function 
fun_vec = @(parameters) objective_vec_HardMixVAF_mex_mcmc_par( ...
        ... % (hyper-)parameters
        parameters, ...
        in.q_case, ...
        in.q, ...
        in.lambda_r, ...
        in.lambda_p, ...
        in.p, ...
        in.n_eval, ...
        ... % FEM results
        data.results.u, ...
        data.results.reaction, ...
        ... % algorithm
        data.algorithm.n_NR_local, ...
        data.algorithm.tol_NR_local, ...
        ... % material
        data.material.CPlaneStress, ...
        data.material.SPlaneStress, ...
        ... % mesh
        data.mesh.n_dof, ...
        data.mesh.n_element, ...
        data.mesh.n_dof_per_element, ...
        data.mesh.element_dof, ...
        data.mesh.Gauss_weights, ...
        data.mesh.n_Gauss_per_dim, ...
        data.mesh.detJ_GP, ...
        data.mesh.B_GP, ...
        ... % bc
        dof_free, ...
        data.bc.dof_reaction ...
        );




%% MCMC Metropolis within Gibbs

% MCMC hyper-parameters

p = 12; % number of parameters
n = n_eval_step*n_residual+1; % number of target data points
d = 12;



nchains = input('Enter number of chains: '); % No. of chains
nSamples = input('Enter number of samples: '); % No. of MCMC samples
nburn = (0.25)*nSamples*p; %------ 25% burn in

verbosity = false; % If verbosity is set to 'true' it will print the MCMC interations


% Initialize all the parameters 
sig2    = 0.01^2; 
noiseBeta = 1/sig2; % noise precision (inverse of variance)


% Initializing the nonlinear function parameters [theta_0, theta_1, ... , 
% theta_7,H_isotropic,H_kinematic]

theta_0 = zeros(1,7);
theta_0(1) = 0.2;
theta_0(2:7) = 0.1./(5.^(1:6));
H_iso_0 = [49,0.45,745];
H_kin_0 = [190,750];
W0 = [theta_0,H_iso_0,H_kin_0];


Z0 = zeros(1,p);%
p0 = 0.1; 
v0 = 1/n;
v1 = 100*v0;
tau2 = 1;

%% Sampling 


% Parallel computing information
% -------------------------------------------------------------------------
n_workers = nchains;
cluster = parcluster('local');
pool = parpool(cluster,n_workers);
% -------------------------------------------------------------------------

results = cell(nchains,1);
tic
parfor i = 1:nchains
    % fprintf('MCMC chain: %d \n',i);
    sig2_i      = 1/(noiseBeta*abs(10*randn));
    tau2_i      = tau2*abs(10*randn);
    p0_i        = p0;
    
    results{i}  = BASAD_Gibbs2(n,d, nSamples, W0, Z0,...
                  sig2_i, tau2_i, p0_i, v0, v1, verbosity,fun_vec);
    fprintf('Parallel chain finished: %d \n',i);
    
end
toc

% shutdown the parallel pool
pool.delete()

dd = p;
% Assembling the results from chains
nsamp = nSamples*dd + 1 - nburn;
ZZ  = nan(nchains*nsamp, dd);
WW  = nan(nchains*nsamp, dd);
sig2= nan(nchains*nsamp, 1);
Wstr= nan(nsamp, dd, nchains);

acceptance_ratio = zeros(nchains,1); 

for i = 1:nchains
    Wstr(:,:,i)  = results{i}.samplesW(nburn+1:end,:);   
    arange       = ((i-1)*nsamp + 1) : i*nsamp;
    ZZ(arange,:) = results{i}.samplesZ(nburn+1:end,:);
    WW(arange,:) = Wstr(:,:,i);
    sig2(arange,:) = results{i}.samplesSig2(nburn+1:end,:);
    acceptance_ratio(i) = results{i}.metro.acceptance_ratio;
end

% Multivariate PSRF (gelman-rubin stat)
Rw = mpsrf_jitter(Wstr);

Zmean = mean(ZZ);
Wmean = mean(WW);
Wcov  = cov(WW);

modelIdx = find(Zmean > 0.5);
% modelIdx = setdiff(modelIdx,1);

% disp(['Wmean: ', num2str(Wmean(modelIdx))])

out.DS = results;
out.ZZ = ZZ;
out.WW = WW;
out.sig2 = mean(sig2);
% out.modelIdx = modelIdx-1;
out.modelIdx = modelIdx;
out.Zmed = Zmean(modelIdx);
out.Wsel = Wmean(modelIdx);
out.Wcov = Wcov(modelIdx, modelIdx);
out.Rw   = Rw;

W_hat = zeros(1,d);
W_hat(modelIdx) = out.Wsel;
% y_hat = f_nl_test(W_hat,X_data);
% sig2_hat = out.sig2;
% M_hat = eye(n)*sig2_hat;
% S_hat = chol(M_hat,'lower');
% y_hat = y_hat + S_hat*randn(n,1);
MSE = mean((fun_vec(W_hat)).^2);
fprintf('MSE: %f',MSE);

% param_names = ['a','b'];
% [~,ax] = plotmatrix(WW);
% iterations = size(ax,1);
% for i = 1:iterations
%     ax(i,1).YLabel.String = param_names(i);
%     ax(iterations,i).XLabel.String = param_names(i);
% end






%% Helper Functions

function DS = BASAD_Gibbs2(n,d,nSamples, W0, Z0, sig2, tau2, p0, v0, v1, Verbosity,fun_vec)

% n: size of target variable 
% d: number of parameters
% nSamples: Total number of chain samples (including burn-in)
% W0    : Initial weight or coefficient vector
% Z0    : Initial latent variable vector

% v1    : fixed multiplier of slab variance
% v0    : fixed multiplier of spike variance
% tau2  : varying scale for the spike and the slab
% sig2  : noise variance
% p0    : inclusion probability

% Deterministic prior parameters
% (a) IG dist (for noise variance sigma^2)
% Values chosen leads to closely non-informative prior IG(0,0)
alpha0      = 1e-4;
beta0       = 1e-4;
% (b) IG dist (for slab scale tau^2)
% degree of freedom nu = 1, s^2 = 1
% Leads to Cauchy(0,1) distribution of the slab
nu = 1; s2 = 1;
alphad0     = nu/2;
betad0      = nu*s2/2;
% (c) Beta dist (for inclusion prob p0)
% Beta(0.1,1) leads to informative prior causing more sparse solutions
a0 = 0.1;
b0 = 1;

% RAM parameters
opt_alpha = 0.234;
gamma = 2/3;



% Perform some initializations
samplesSig2 = zeros(nSamples*d + 1, 1);   % Sigma^2 (noise variance)
samplesP    = zeros(nSamples*d + 1, 1);   % P0 (inclusion probability)
samplesTau2 = zeros(nSamples*d + 1, d);   % Tau^2 (predictor-specific slab scale variance)
samplesZ    = zeros(nSamples*d + 1, d);
samplesW    = zeros(nSamples*d + 1, d);

% Proposal_standard_deviation
prop_sigma = zeros(nSamples*d + 1,d);

samplesSig2(1)  = sig2;
samplesP(1)     = p0;
samplesTau2(1,:)= tau2*ones(1,d);     
samplesZ(1,:)   = Z0;
samplesW(1,:)   = W0;

% GG      = X'*X;
% tmu     = X'*y;
% eyed    = eye(d);
zerosd  = zeros(1,d);

% Proposal candidate covariance matrix 
sig_prop_vect = [0.001,ones(1,6).*0.001,[1,0.01,1,1,1]];

prop_sigma(1,:) = sig_prop_vect;


% M_prop = diag([ones(1,7).*(sig_prop^2),([0.8,0.01,0.8,0.8,0.8]).^2]);
% M_prop = eyed.*(sig_prop^2);
% M_prop = diag([sig_prop_1,sig_prop_2,sig_prop_3].^2);

accepted = 0;

i = 2;
% Start sampling from the conditional distributions
for k = 1:nSamples
    for j = 1:d
    
        prevZ       = samplesZ(i-1,:);
        prevT       = samplesTau2(i-1,:);
        prevP       = samplesP(i-1);
        prevsig2    = samplesSig2(i-1);
        prevW       = samplesW(i-1,:);
        
        % Update the weights W -----here W has been sample using metropolis
        % algorithm
    
        D  = diag( 1./(v1 * (prevZ.*prevT) + v0 * ((1-prevZ).*prevT)));

%         prevW_log = log(prevW);
%         
%         W_prop_log_j = prevW_log(j) + sig_prop_vect(j)*nn;
%         W_prop_log = prevW_log;
%         W_prop_log(j) = W_prop_log_j;
%         W_prop_exp = exp(W_prop_log);
        W_prop = prevW;
        W_prop(j) = prevW(j) + sig_prop_vect(j)*randn;
%         W_prop(j) = W_prop_j;
    %     w1 = 0;
        if ~(W_prop(1)>sum(abs(W_prop(2:7)))) 
            admissible = false;
            fprintf('Entered perturbation loop.\n')
            while ~admissible
    %             parameters_proposal(1:end-5) = parameters_proposal(1:end-5) + perturbation_guess_theta'.*randn(in.n_feature,1);
    %             order = check_order(parameters_proposal(1));
                
                W_prop(2:7) = W_prop(2:7)./(2.^(1:6));
                if W_prop(1)>sum(abs(W_prop(2:7)))
                    admissible = true;
                end
    %             w1 = w1+1;
            end
    
        end
    
        err_prop = fun_vec(W_prop);
        err = fun_vec(prevW); 
        
        % Likelihood ratio

        Likelihood_ratio =  - ((err_prop'*err_prop)/(2*prevsig2)) + (err'*err)/(2*prevsig2);
        Prior_ratio = - ((W_prop*D*W_prop')/2) + (prevW*D*prevW')/2;
        L_P_ratio = Likelihood_ratio + Prior_ratio;


        AA = exp(L_P_ratio);

    
        alpha = min(1,AA);
    
       
        if alpha >= rand && ~isnan(AA)
            newW = W_prop;
            accepted = accepted+1;
        else
            newW = samplesW(i-1,:);
        end
        samplesW(i,:) = newW;

        eta = min(1, 1 * i^(-gamma));
        % fac = (eye(1) + eta * (alpha - opt_alpha) * (nn * nn') / norm(nn)^2);
        % M = sig_prop_vect(j) * fac * sig_prop_vect(j)' ;
        % sig_prop_vect(j) = chol(M, 'lower');
        sig_prop_vect(j) = sqrt((sig_prop_vect(j)^2)*(1 + eta*(alpha - opt_alpha)));

        prop_sigma(i,:) = sig_prop_vect;
    
        % Update the latent variables Z
        term0           = (1-prevP) * mydnorm(newW, zerosd, v0*prevT);
        term1           = prevP * mydnorm(newW, zerosd, v1*prevT);
        pratio          = term1./(term0 + term1);
        newZ            = rand(1,d) < pratio;
        newZ(1)         = 1;
        samplesZ(i,:)   = newZ;
    %     disp(newZ)
            
        % Update sigma^2
    %     e               = fun_vec(newW); % this increases the computation time
    
        if alpha >= rand && ~isnan(AA)
            e = err_prop;
        else
            e = err;
        end
        
    
        alpha           = alpha0 + 0.5*n ;
        beta            = beta0 + 0.5*(e'*e) ;
        newsig2         = 1/gamrnd(alpha, 1/beta); 
        samplesSig2(i)  = newsig2;
        
        % Update tau^2 (each coefficient has a separate tau)
        T1              = v1*newZ + v0*(1-newZ);
        alphad          = alphad0 + 0.5;
        betad           = betad0 + 0.5*(newW.^2) ./ (T1);
    %     newtau2         = 1./gamrnd(alphad, 1./betad);
    %     samplesTau2(i,:)= newtau2;
        for jj = 1:d
            newtau2     = 1/gamrnd(alphad, 1/betad(jj));
            samplesTau2(i,jj)= newtau2;
        end
        
        % Update the P from beta distribution
        sz              = sum(newZ);
        samplesP(i)     = betarnd(a0 + sz, b0 + d - sz);
        
        if(Verbosity)
            if(mod(i-1,100)==0)
                fprintf('     %d samples\n',i-1);
            end
        end

        i = i + 1;

    end

end

DS.samplesW    = samplesW;
DS.samplesZ    = samplesZ;
DS.samplesP    = samplesP;
DS.samplesSig2 = samplesSig2;
DS.samplesTau2 = samplesTau2;

DS.prior.sig2  = [alpha0, beta0];
DS.prior.tau2  = [alphad0, betad0];
DS.prior.p0    = [a0, b0];

DS.metro.acceptance_ratio = accepted/(nSamples*d);
DS.prop_sigma = prop_sigma;
    
end


function pdfxx = mydnorm(W, mu, sigsq)

pdfxx = (1./sqrt(2*pi*sigsq)) .* exp(-((W-mu).^2)./(2*sigsq));

end
