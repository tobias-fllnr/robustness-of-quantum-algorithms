clear all
clc
rng(0);

% toggle between systematic (=1) and independent (=0) errors
SYSTEMATIC = 1;
format long


% Unitary operations
n = 1;  % Number of qubits
% Draw noise samples
N_samples = 2;


X = [0 1;1 0];
Y = [0 -1i;1i 0];
Z = [1 0;0 -1];

theta = pi/4;


scale = 1:10;
noise_bounds = [1e-4*scale 1e-3*scale 1e-2*scale];
n_q = length(noise_bounds);
q = 0;

fidelities_worst_store = zeros(n_q,20);
fidelity_bounds_store = zeros(n_q,20);
gammas_store = zeros(n_q,20);

for noise_bound = noise_bounds
    noise_bound 

    q = q+1;


% Nominal circuit
circuit_1 = cell(1, 1);
error_hamiltonians_1 = cell(1,1);
[circuit_1{1},error_hamiltonians_1{1}]  = rotation_u(theta,0,noise_bound);

% Pulses from kabytayev2014robustness
Phi_1 = acos(-theta/(4*pi));
k = asin(sin(theta/2)/2);

circuits = {};
error_hamiltonians = {};
circuits{1} = circuit_1;
error_hamiltonians{1} = error_hamiltonians_1;

jones2003robust = 1;
SK1 = 0;
BB1 = 0;
CORPSE = 0;
rCinSK = 0;
rCinBB = 0;
DESIGNED = 1;
if jones2003robust
    circuit_2 = cell(1,5);
    error_hamiltonians_2 = cell(1,5);
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(theta/2,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(pi,Phi_1,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(2*pi,3*Phi_1,noise_bound);
    [circuit_2{4}, error_hamiltonians_2{4}] = rotation_u(pi,Phi_1,noise_bound);
    [circuit_2{5}, error_hamiltonians_2{5}] = rotation_u(theta/2,0,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if SK1
    circuit_2 = cell(1,3);
    error_hamiltonians_2 = cell(1,3);    
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(theta,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(2*pi,-Phi_1,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(2*pi,Phi_1,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if BB1
    circuit_2 = cell(1,4);
    error_hamiltonians_2 = cell(1,4);    
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(theta,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(pi,Phi_1,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(2*pi,3*Phi_1,noise_bound);
    [circuit_2{4}, error_hamiltonians_2{4}] = rotation_u(pi,Phi_1,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if CORPSE
    circuit_2 = cell(1,3);
    error_hamiltonians_2 = cell(1,3);    
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(2*pi+theta/2-k,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(2*pi-2*k,pi,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(theta/2-k,0,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if rCinSK
    circuit_2 = cell(1,5);
    error_hamiltonians_2 = cell(1,5);    
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(2*pi+theta/2-k,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(2*pi-2*k,pi,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(theta/2-k,0,noise_bound);
    [circuit_2{4}, error_hamiltonians_2{4}] = rotation_u(2*pi,-Phi_1,noise_bound);
    [circuit_2{5}, error_hamiltonians_2{5}] = rotation_u(2*pi,Phi_1,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if rCinBB
    circuit_2 = cell(1,6);
    error_hamiltonians_2 = cell(1,6);    
    [circuit_2{1}, error_hamiltonians_2{1}] = rotation_u(2*pi+theta/2-k,0,noise_bound);
    [circuit_2{2}, error_hamiltonians_2{2}] = rotation_u(2*pi-2*k,pi,noise_bound);
    [circuit_2{3}, error_hamiltonians_2{3}] = rotation_u(theta/2-k,0,noise_bound);
    [circuit_2{4}, error_hamiltonians_2{4}] = rotation_u(pi,Phi_1,noise_bound);
    [circuit_2{5}, error_hamiltonians_2{5}] = rotation_u(2*pi,3*Phi_1,noise_bound);
    [circuit_2{6}, error_hamiltonians_2{6}] = rotation_u(pi,Phi_1,noise_bound);
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end
if DESIGNED
    load('Design of new composite pulses/designed_angles.mat');
    depth = length(angles_opt)/2;
    circuit_2 = cell(1,depth);
    error_hamiltonians_2 = cell(1,depth); 
    for j = 1:depth
        [circuit_2{j}, error_hamiltonians_2{j}] = rotation_u(angles_opt(2*j-1),angles_opt(2*j),noise_bound);
    end
    circuits{end+1} = circuit_2;
    error_hamiltonians{end+1} = error_hamiltonians_2;
end

errors = error_hamiltonians;

% Compute Lipschitz bounds
lipschitz = {};
lipschitz{1} = norm(errors{1}{1}/noise_bound);
for j = 2:size(circuits,2)
    lipschitz{j} = 0;
    for k = 1:size(errors{j},2)
        lipschitz{j} = lipschitz{j}+norm(errors{j}{k}/noise_bound);
    end
end

if 0 % set to 1 for different error types
    %% Errors
    % X
    %H_e = noise_bound*[0 1;1 0];
    % Y
    %H_e = noise_bound*[0 -1i;1i 0];
    % Z
    H_e = noise_bound*[1 0;0 -1];

    errors = {};
    for j = 1:size(circuits,2)
        error = {};
        for k = 1:size(circuits{j},2)
            error{end+1} = H_e;
        end
        errors{end+1} = error;
    end
end




%% some basic parameters
depth = {};
depth_list = [];
ells = {};
delta = {};
num_circuits = size(circuits,2);
for j = 1:num_circuits
    depth{j} = size(circuits{j},2);
    depth_list(j) = depth{j};
    ells{j} = ones(depth{j},1);
    delta{j} = -inf;
    for k = 1:size(errors{j},2)
        bound_new = norm(errors{j}{k});
        if bound_new>delta{j}
            delta{j} = bound_new;
        end
    end
end


%% Compute full circuit unitaries
unitaries = {};
fidelities_unitaries = [];
for j = 1:num_circuits
    unitaries{j} = simulation(circuits{j});
    fidelities_unitaries = [fidelities_unitaries fidelity(unitaries{j},unitaries{1})];
end
fidelities_unitaries

%% Compute averaged interaction Hamiltonian norm and fidelity bound
gammas = zeros(1,num_circuits);
fidelity_bounds = zeros(1,num_circuits);

for j = 1:num_circuits
    [gammas(j), fidelity_bounds(j)] = averaged_hamiltonian(circuits{j},errors{j},delta{j},ells{j},SYSTEMATIC); 
end

%% Simulate noisy circuits

if SYSTEMATIC
    % systematic errors
    noise_samples = 2*(ones(max(depth_list),1,N_samples)-0.5);
else
    % independent errors
    noise_samples = 2*(rand(max(depth_list),1,N_samples)-0.5);
end

fidelities_store = zeros(N_samples,2);

for j = 1:num_circuits
    for k = 1:N_samples
        U_noisy = simulation_noisy(circuits{j},errors{j},noise_samples(:,:,k),ells{j});
        fidelities_store(k,j) = fidelity(U_noisy,unitaries{j});
    end
end

fidelities_worst = min(fidelities_store);
%fidelities_mean = mean(fidelities_store,1)
fidelity_bounds;
gammas;


fidelities_worst_store(q,1:num_circuits) = fidelities_worst;
fidelity_bounds_store(q,1:num_circuits) = fidelity_bounds;
gammas_store(q,1:num_circuits) = gammas;
end

fidelities_worst_store = fidelities_worst_store(:,1:num_circuits);
fidelity_bounds_store = fidelity_bounds_store(:,1:num_circuits);
gammas_store = gammas_store(:,1:num_circuits);

lipschitz_bounds = [];
fidelity_bounds_lipschitz_store = zeros(n_q,num_circuits);
for j = 1:num_circuits
    lipschitz_bounds = [lipschitz_bounds lipschitz{j}];
    for q = 1:n_q
        fidelity_bounds_lipschitz_store(q,j) = 1-lipschitz_bounds(j)^2*noise_bounds(q)^2;
    end
end



%% Plot the results


%%
if 1
figure

loglog(noise_bounds,1-fidelity_bounds_store(:,1),'r--x','LineWidth',3)
hold on
grid on
loglog(noise_bounds,1-fidelity_bounds_store(:,2),'b:o','LineWidth',3)
loglog(noise_bounds,1-fidelity_bounds_store(:,3),'k-*','LineWidth',3)


xlabel('Noise level $\delta$','interpreter','latex','FontSize',12)
ylabel('Infidelity $1-F_{\mathrm{wc}}$','interpreter','latex','FontSize',12)
legend('$R_X(\frac{\pi}{4})$','Pulses by Jones','Our pulses','Location','NorthWest','interpreter','latex','FontSize',12)
end


%%

function fid = fidelity(U1,U2)
    n = size(U1,1);
    fid = abs(trace(U1'*U2)/n)^2;
end

function [gamma, fidelity_bound] = averaged_hamiltonian(circuit,error,delta,ell,SYSTEMATIC)
    len = size(circuit);
    depth = len(2);
    n = size(circuit{1});
    n = n(1);
    
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = circuit{1};
    for k = 3:depth
        Vk = circuit{1};
        for i = 2:(k-1)
            Vk = circuit{i} * Vk;
        end
        V{k} = Vk;
    end
    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        if iscell(error{j})
            for k = 1:ell(j)
                M{j}{k} = V{j}'*error{j}{k}*V{j}/ell(j);
            end
        else
                M{j}{1} = V{j}'*error{j}*V{j};
        end
    end

    % Compute the concatenated A
    M_con = [];
    for j = 1:depth
        for k = 1:ell(j)
            Mkj = reshape(M{j}{k},[], 1)/depth;
            M_con = [M_con, Mkj];
        end
    end

    %gamma = compute_gamma_fmincon(M,ell);
    gamma = compute_gamma_sigma_exact(M,ell,SYSTEMATIC)/delta;
    %gamma = norm(M_con)*sqrt(depth)/delta;
    % systematic norm-based bound
    %gamma = norm(M_con*ones(depth,1));
    
    if SYSTEMATIC
        % Commutator-based bound from the theorem
        R_diff = 0;
        for j = 1:depth
            mat_sum = zeros(2,2);
            for k = j+1:depth
                mat_sum = mat_sum+M{j}{1}*M{k}{1}-M{k}{1}*M{j}{1};
            end
            R_diff = R_diff+0.5*norm(mat_sum);
        end
        fidelity_bound = 1-(R_diff+depth*delta*gamma)^2;
    else
        fidelity_bound = 1-delta^2*depth^2*(delta*(depth-1)/2+gamma)^2;
    end

    % Upper bound on commutator-based bound from the theorem
    % (use the triangle inequality)
    if 0
        R_diff = 0;
        for j = 1:depth
            for k = j+1:depth
                R_diff = R_diff+0.5*norm(M{j}{1}*M{k}{1}-M{k}{1}*M{j}{1});
            end
        end
        
        fidelity_bound = 1-(R_diff+depth*gamma)^2;
    end
    
    % Exponential difference from the proof
    if 0
        term_1 = eye(2);
        for j = 1:depth
            term_1 = expm(-1i*M{j}{1})*term_1;
        end
        term_2 = zeros(2,2);
        for j = 1:depth
            term_2 = term_2 + M{j}{1};
        end
        term_2 = expm(-1i*term_2);
        fidelity_bound = 1-(norm(term_1-term_2)+depth*gamma)^2;
    end

    % full fidelity bound via fmincon
    %gamma = 0;
    %fidelity_bound = compute_fidelity_bound_fmincon(M,delta,ell);
end

function U_output = simulation(circuit)
    n = size(circuit{1});
    U_output = eye(n(1));
    len = size(circuit);
    depth = len(2);
    for i = 1:depth
        U_output = circuit{i}*U_output;
    end
end

function U_output = simulation_noisy(circuit,error,noise_samples,ell)
    n = size(circuit{1});
    n = n(1);
    U_output = eye(n);

    len = size(circuit);
    depth = len(2);
    for i = 1:depth
        if iscell(error{i})
            for k = 1:ell(i)
                H_e = zeros(n,n);
                H_e = H_e + (1/ell(i))*noise_samples(i,k)*error{i}{k};
            end
        else
            H_e = noise_samples(i,1)*error{i};
        end
        
        U_output = circuit{i}*expm(-1i*H_e)*U_output;
    end
end

function gamma = compute_gamma_fmincon(M,ell)
    options = optimoptions('fmincon','MaxFunctionEvaluations',1e6,'MaxIterations',1e6,'StepTolerance',1e-12);
    obj = @(theta) -compute_gamma_cost(theta,M,ell);
    ell_new = [];
    for j = 1:length(ell)
        ell_new = [ell_new;ell(j)];
        if ell(j) == 2
            ell_new = [ell_new;ell(j)]; 
        end
    end
    lb = -((ell_new==1)+0.5*(ell_new~=1));
    ub = ((ell_new==1)+0.5*(ell_new~=1));
    A_eq = zeros(sum(ell)-1,sum(ell));
    b_eq = zeros(sum(ell)-1,1);
    % systematic errors
    if 1
        for j = 1:sum(ell)-1
            A_eq(j,j) = 1;
            A_eq(j,j+1) = -1;
        end
    end
    [theta, gamma] = fmincon(obj,rand(sum(ell),1)/2,[],[],A_eq,b_eq,lb,ub,[],options);
    gamma = -gamma;
end

function cost = compute_gamma_cost(theta,M,ell)
    len = size(M);
    depth = len(2);
    n = size(M,1);
    mat = zeros(n,n);
    
    for j = 0:depth-1
        for k = 1:ell(j+1)
            mat = mat + theta(sum(ell(1:j))+k)*M{j+1}{k};
        end
    end
    mat = mat/depth;
    cost = norm(mat);
end

function fidelity_bound = compute_fidelity_bound_fmincon(M,delta,ell)
    options = optimoptions('fmincon','MaxFunctionEvaluations',1e9,'MaxIterations',1e6,'StepTolerance',1e-12);
    obj = @(theta) -compute_fidelity_bound_cost(theta,M,ell);
    ell_new = [];
    for j = 1:length(ell)
        ell_new = [ell_new;ell(j)];
        if ell(j) == 2
            ell_new = [ell_new;ell(j)]; 
        end
    end
    lb = -delta*((ell_new==1)+0.5*(ell_new~=1));
    ub = delta*((ell_new==1)+0.5*(ell_new~=1));

    A_eq = zeros(sum(ell)-1,sum(ell));
    b_eq = zeros(sum(ell)-1,1);
    % systematic errors
    if 1
        for j = 1:sum(ell)-1
            A_eq(j,j) = 1;
            A_eq(j,j+1) = -1;
        end
    end

    [theta, f] = fmincon(obj,delta*rand(sum(ell),1)/2,[],[],A_eq,b_eq,lb,ub,[],options);
    f = -f;
    fidelity_bound = 1-f;
end

function cost = compute_fidelity_bound_cost(theta,M,ell)
    len = size(M);
    depth = len(2);
    n = size(M,1);

    G = {};
    for j = 0:depth-1
        G{j+1} = zeros(n,n);
        for k = 1:ell(j+1)
            G{j+1} = G{j+1}+theta(sum(ell(1:j))+k)*M{j+1}{k};
        end
    end

    term1 = 0;
    for j=1:depth
        sum_mat = zeros(n,n);
        for k=j+1:depth
            sum_mat = sum_mat + G{j}*G{k}-G{k}*G{j};
        end
        term1 = term1 + 0.5*norm(sum_mat);
    end

    mat2 = zeros(n,n);
    for j = 1:depth
        mat2 = mat2 + G{j};
    end
    cost = (0.5*term1+norm(mat2))^2;
end

function gamma = compute_gamma_sigma_exact(M,ell,SYSTEMATIC)
    len = size(M);
    depth = len(2);
    n = size(M,1);

    ell_new = [];
    for j = 1:depth
        ell_new = [ell_new;ell(j)];
        if ell(j) == 2
            ell_new = [ell_new;ell(j)]; 
        end
    end

    % Systematic errors
    if SYSTEMATIC
        theta_vertex = ones(depth,1);
        mat = zeros(n,n);
        for j = 1:depth
            mat = mat + theta_vertex(j)*M{j}{1};
        end
        gamma = norm(mat/depth);
    else
    % Independent errors
        gamma = -inf;
        for j = 1:2^(sum(ell))
            if mod(j,100)==0
                j
            end
            indices = decimalToBinaryVector(j);
            indices = [zeros(1,sum(ell)-length(indices)) indices];
            if j == 2^(sum(ell))
                indices = zeros(1,depth);
            end
            theta_vertex = ones(depth,1).*(-1).^(indices');
            mat = zeros(n,n);
            for j = 0:depth-1
                for k = 1:ell(j+1)
                    mat = mat + theta_vertex(sum(ell(1:j))+k)*M{j+1}{k};
                end
            end
            mat = mat/depth;
            if norm(mat)>gamma
                gamma = norm(mat);
            end
        end
    end
end


function [gamma, fidelity_bound] = averaged_hamiltonian_cutting(circuit,error,delta,ell)
% compute gamma via cutting
    len = size(circuit);
    depth = len(2);

    %cuts = [19 19 19];
    cuts = depth;
    %cuts = [10 10 10 10 10 7];
    cuts_cum = [0 cumsum(cuts)];
    cut_num = length(cuts);

    for j = 1:cut_num
        for k = 1:cuts(j)
            circuit_reduced{j}{k} = circuit{cuts_cum(j)+k};
            error_reduced{j}{k} = error{cuts_cum(j)+k};
        end
        ell_reduced{j} = ell(cuts_cum(j)+1:cuts_cum(j)+cuts(j));
    end

    gammas_cut = zeros(cut_num,1);
    for j = 1:cut_num
        % computing fidelity bound via gamma
        [gammas_cut(j), fidelity_temp] = averaged_hamiltonian(circuit_reduced{j},error_reduced{j},delta,ell_reduced{j});
        % computing fidelity bound directly using fmincon
        %[gammas_cut(j), fidelity_bound] = averaged_hamiltonian(circuit_reduced{j},error_reduced{j},delta,ell_reduced{j});
        gammas_cut(j) = gammas_cut(j)*cuts(j);
    end

    gamma = sum(gammas_cut)/depth;
    % comment out the following when computing directly f
    fidelity_bound = 1-delta^2*depth^2*(delta*(depth-1)/2+gamma)^2;
end



function [gate, error_hamiltonian] = rotation_u(theta,phi,noise_bound)
    X = [0 1;1 0];
    Y = [0 -1i;1i 0];
    Z = [1 0;0 -1];
    n_x = cos(phi);
    n_y = sin(phi);
    n_z = 0;
    %gate = cos(angle/2)*eye(2)-1i*sin(angle/2)*(n_x*X+n_y*Y+n_z*Z);
    hamiltonian = 0.5*theta*(n_x*X+n_y*Y+n_z*Z);
    % normalize the Hamiltonian
    hamiltonian = hamiltonian-0.5*eye(2)*(max(eig(hamiltonian))+min(eig(hamiltonian)));
    gate = expm(-1i*hamiltonian);
    error_hamiltonian = noise_bound*hamiltonian;
end