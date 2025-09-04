clear all
clc
rng(0);

load('composite_pulses_circuits.mat')

%% Parameters
noise_bound = 0.05;
% ideal quantum gate
theta = pi/4;
[U_ideal,E_ideal]  = rotation_u(theta,0,noise_bound);

% [Jones, 2003]
angles_init = zeros(5,1);
angles_init(1) = theta/2;
angles_init(2) = 0;
angles_init(3) = pi;
angles_init(4) = Phi_1;
angles_init(5) = 2*pi;
angles_init(6) = 3*Phi_1;
angles_init(7) = pi;
angles_init(8) = Phi_1;
angles_init(9) = theta/2;
angles_init(10) = 0;

depth = length(angles_init)/2;

%% Set up fmincon
options = optimoptions('fmincon','MaxFunctionEvaluations',1e6,'MaxIterations',1e6,'StepTolerance',1e-12);

% weighting parameters for cost function
weight_indep = 1;
weight_Rx = 0;
weight_Ry = 0;
weight_Rz = 0;

% weighted multiobjective cost function
obj = @(angles) weight_indep*infidelity_cost(angles,noise_bound) + weight_Rx*infidelity_cost_Rx(angles,noise_bound) + weight_Ry*infidelity_cost_Ry(angles,noise_bound) + weight_Rz*infidelity_cost_Rz(angles,noise_bound);

% constraints for optimization
lb = -2*pi*ones(2*depth,1);
ub = 2*pi*ones(2*depth,1);
A_eq = [];
b_eq = [];

% nonlinear constraint
nlcon = @(angles) nonlcon(angles,noise_bound,U_ideal);


[angles_opt, R_weighted] = fmincon(obj,angles_init,[],[],A_eq,b_eq,lb,ub,nlcon,options);

angles_opt

R_weighted

%% check constraint satisfaction
[c, c_eq] = nonlcon(angles_opt,noise_bound,U_ideal)


%%
save("designed_angles.mat","angles_opt")


function [c, c_eq] = nonlcon(angles,noise_bound,U_ideal)
    %% equality constraint
    n = 2;
    % Define parameterized gates and Hermitian error operators
    depth = length(angles)/2;
    U = cell(1,depth);
    H = cell(1,depth);
    for j = 1:depth
        [U{j}, H{j}] = rotation_u(angles(2*j-1),angles(2*j),noise_bound);
    end
    % Compute unitary matrix
    U_output = eye(n);
    for j = 1:depth
        U_output = U{j}*U_output;
    end

    % compute equality constraint
    c_eq = 1-abs(trace(U_output'*U_ideal)/n)^2;

    %% inequality constraint
    % define noise level
    delta = -inf;
    for j =1:depth
        bound_new = norm(H{j});
        if bound_new > delta
            delta = bound_new;
        end
    end

    % Compute matrices relevant for bound
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = U{1};
    for k = 3:depth
        Vk = U{1};
        for i = 2:(k-1)
            Vk = U{i} * Vk;
        end
        V{k} = Vk;
    end    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        M{j} = V{j}'*H{j}*V{j};
    end    

    theta_vertex = ones(depth,1);
    mat = zeros(n,n);
    for j = 1:depth
        mat = mat + theta_vertex(j)*M{j};
    end
    gamma_sys = norm(mat/depth)/delta;

    R_diff = 0;
    for j = 1:depth
        mat_sum = zeros(2,2);
        for k = j+1:depth
            mat_sum = mat_sum+M{j}*M{k}-M{k}*M{j};
        end
        R_diff = R_diff+0.5*norm(mat_sum);
    end
    R_sys = (R_diff+depth*delta*gamma_sys)^2;
    % The following ensures a fidelity of at least 99.999% for systematic
    % errors
    c = R_sys-(1-0.999995);
end

function R = infidelity_cost(angles,noise_bound)
    n = 2;
    %% define parameterized gates and Hermitian error operators
    depth = length(angles)/2;
    U = cell(depth,1);
    H = cell(depth,1);
    for j = 1:depth
        [U{j}, H{j}] = rotation_u(angles(2*j-1),angles(2*j),noise_bound);
    end
    % define noise level
    delta = -inf;
    for j =1:depth
        bound_new = norm(H{j});
        if bound_new > delta
            delta = bound_new;
        end
    end

    %% Compute matrices relevant for bound
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = U{1};
    for k = 3:depth
        Vk = U{1};
        for i = 2:(k-1)
            Vk = U{i} * Vk;
        end
        V{k} = Vk;
    end    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        M{j} = V{j}'*H{j}*V{j};
    end    

    %% Compute independent infidelity bound
    gamma_ind = -inf;
    for j = 1:2^depth
        indices = decimalToBinaryVector(j);
        indices = [zeros(1,depth-length(indices)) indices];
        if j == 2^depth
            indices = zeros(1,depth);
        end
        theta_vertex = ones(depth,1).*(-1).^(indices');
        mat = zeros(n,n);
        for j = 1:depth
            mat = mat + theta_vertex(j)*M{j};
        end
        mat = mat/depth;
        if norm(mat)>gamma_ind
            gamma_ind = norm(mat);
        end
    end
    gamma_ind = gamma_ind/delta;

    R = delta^2*depth^2*(delta*(depth-1)/2+gamma_ind)^2;
end


function R = infidelity_cost_Rx(angles,noise_bound)
    n = 2;
    %% define parameterized gates and Hermitian error operators
    depth = length(angles)/2;
    U = cell(depth,1);
    H = cell(depth,1);
    for j = 1:depth
        [U{j}, H{j}] = rotation_u(angles(2*j-1),angles(2*j),noise_bound);
        H{j} = noise_bound*[0 1;1 0];
    end
    % define noise level
    delta = -inf;
    for j =1:depth
        bound_new = norm(H{j});
        if bound_new > delta
            delta = bound_new;
        end
    end

    %% Compute matrices relevant for bound
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = U{1};
    for k = 3:depth
        Vk = U{1};
        for i = 2:(k-1)
            Vk = U{i} * Vk;
        end
        V{k} = Vk;
    end    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        M{j} = V{j}'*H{j}*V{j};
    end    

    %% Compute independent infidelity bound
    gamma_ind = -inf;
    for j = 1:2^depth
        indices = decimalToBinaryVector(j);
        indices = [zeros(1,depth-length(indices)) indices];
        if j == 2^depth
            indices = zeros(1,depth);
        end
        theta_vertex = ones(depth,1).*(-1).^(indices');
        mat = zeros(n,n);
        for j = 1:depth
            mat = mat + theta_vertex(j)*M{j};
        end
        mat = mat/depth;
        if norm(mat)>gamma_ind
            gamma_ind = norm(mat);
        end
    end
    gamma_ind = gamma_ind/delta;

    R = delta^2*depth^2*(delta*(depth-1)/2+gamma_ind)^2;
end



function R = infidelity_cost_Ry(angles,noise_bound)
    n = 2;
    %% define parameterized gates and Hermitian error operators
    depth = length(angles)/2;
    U = cell(depth,1);
    H = cell(depth,1);
    for j = 1:depth
        [U{j}, H{j}] = rotation_u(angles(2*j-1),angles(2*j),noise_bound);
        H{j} = noise_bound*[0 -1i;1i 0];
    end
    % define noise level
    delta = -inf;
    for j =1:depth
        bound_new = norm(H{j});
        if bound_new > delta
            delta = bound_new;
        end
    end

    %% Compute matrices relevant for bound
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = U{1};
    for k = 3:depth
        Vk = U{1};
        for i = 2:(k-1)
            Vk = U{i} * Vk;
        end
        V{k} = Vk;
    end    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        M{j} = V{j}'*H{j}*V{j};
    end    

    %% Compute independent infidelity bound
    gamma_ind = -inf;
    for j = 1:2^depth
        indices = decimalToBinaryVector(j);
        indices = [zeros(1,depth-length(indices)) indices];
        if j == 2^depth
            indices = zeros(1,depth);
        end
        theta_vertex = ones(depth,1).*(-1).^(indices');
        mat = zeros(n,n);
        for j = 1:depth
            mat = mat + theta_vertex(j)*M{j};
        end
        mat = mat/depth;
        if norm(mat)>gamma_ind
            gamma_ind = norm(mat);
        end
    end
    gamma_ind = gamma_ind/delta;

    R = delta^2*depth^2*(delta*(depth-1)/2+gamma_ind)^2;
end



function R = infidelity_cost_Rz(angles,noise_bound)
    n = 2;
    %% define parameterized gates and Hermitian error operators
    depth = length(angles)/2;
    U = cell(depth,1);
    H = cell(depth,1);
    for j = 1:depth
        [U{j}, H{j}] = rotation_u(angles(2*j-1),angles(2*j),noise_bound);
        H{j} = noise_bound*[1 0;0 -1];
    end
    % define noise level
    delta = -inf;
    for j =1:depth
        bound_new = norm(H{j});
        if bound_new > delta
            delta = bound_new;
        end
    end

    %% Compute matrices relevant for bound
    % Compute the V_j's
    V = {};
    V{1} = eye(n);
    V{2} = U{1};
    for k = 3:depth
        Vk = U{1};
        for i = 2:(k-1)
            Vk = U{i} * Vk;
        end
        V{k} = Vk;
    end    
    % Compute the M_j's
    M = {};
    for j = 1:depth
        M{j} = V{j}'*H{j}*V{j};
    end    

    %% Compute independent infidelity bound
    gamma_ind = -inf;
    for j = 1:2^depth
        indices = decimalToBinaryVector(j);
        indices = [zeros(1,depth-length(indices)) indices];
        if j == 2^depth
            indices = zeros(1,depth);
        end
        theta_vertex = ones(depth,1).*(-1).^(indices');
        mat = zeros(n,n);
        for j = 1:depth
            mat = mat + theta_vertex(j)*M{j};
        end
        mat = mat/depth;
        if norm(mat)>gamma_ind
            gamma_ind = norm(mat);
        end
    end
    gamma_ind = gamma_ind/delta;

    R = delta^2*depth^2*(delta*(depth-1)/2+gamma_ind)^2;
end

function [gate, error_hamiltonian] = rotation_u(theta,phi,noise_bound)
    X = [0 1;1 0];
    Y = [0 -1i;1i 0];
    Z = [1 0;0 -1];
    n_x = cos(phi);
    n_y = sin(phi);
    n_z = 0;
    hamiltonian = 0.5*theta*(n_x*X+n_y*Y+n_z*Z);
    % normalize the Hamiltonian
    hamiltonian = hamiltonian-0.5*eye(2)*(max(eig(hamiltonian))+min(eig(hamiltonian)));
    gate = expm(-1i*hamiltonian);
    error_hamiltonian = noise_bound*hamiltonian;
end