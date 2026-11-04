%% Lacet (rotation sur soi-même) : données sur le simulateur UAV_model.slx
%  Le modèle RNN (u -> az) ne connaît pas le lacet : les données fournies n'ont qu'une commande,
%  envoyée aux 4 moteurs, qui ne crée aucun couple de lacet. Ce script :
%   1. cherche la commande de vol stationnaire u_h
%   2. essaie les 3 façons d'appliquer une commande différentielle δ aux 4 moteurs
%      (u_i = u + s_i·δ, deux moteurs à +δ, deux à -δ) et garde celle qui fait tourner le drone
%      (wz) sans le faire basculer (wx, wy) ; le signe est choisi pour que δ > 0 donne wz > 0
%   3. génère N_VOLS vols avec u et δ aléatoires (paliers), pour identifier δ -> w_dot_z
%  Écrit : tests_lacet.mat (étape 2, sans bruit), donnees_lacet.mat (étape 3),
%          figures/test_repartition.png, figures/exemple_vol.png
%  Le compte rendu (schémas, analyse, modèle) est dans lacet.ipynb.
%  Le harness (UAV_model + entrées/sorties) est construit en mémoire, il n'est pas enregistré.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', 'Matlab files'));
if ~isfolder('figures'), mkdir('figures'); end

Ts = 0.05;           % pas de temps imposé [s]
G_NOISE = 0.01;      % bruit des données de l'étape 3 (valeur de random_init) ; 0 aux étapes 1 et 2
N_VOLS = 100;        % nombre de vols générés
T_VOL = 20;          % durée d'un vol [s]
DELTA_TEST = 0.05;   % commande différentielle de l'étape 2
DELTA_MAX = 0.05;    % amplitude max de δ dans les données (δ = 0.05 donne déjà ~8.6 rad/s² en stationnaire)
HARNESS = 'lacet_harness';
NOMS = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'wx', 'wy', 'wz', ...
        'ax', 'ay', 'az', 'w_dot_x', 'w_dot_y', 'w_dot_z'};    % sorties 1 à 18 de UAV_model
c = @(nom) find(strcmp(NOMS, nom));                            % colonne d'une sortie

construire_harness(HARNESS);


%% 1. Vol stationnaire : u_h tel que az = 0 (méthode de la sécante, sans bruit)
az_moyen = @(u) mean(simuler(HARNESS, u * ones(round(2 / Ts) + 1, 4), Ts, 0, ...
                             0.8, 1.2, c('az')));             % après le filtre moteur (0.2 s)
u_a = 0.4; f_a = az_moyen(u_a);
u_b = 0.6; f_b = az_moyen(u_b);
for i = 1:8
    u_h = u_b - f_b * (u_b - u_a) / (f_b - f_a);
    f_h = az_moyen(u_h);
    if abs(f_h) < 1e-3, break; end
    u_a = u_b; f_a = f_b; u_b = u_h; f_b = f_h;
end
fprintf('1. Vol stationnaire : u_h = %.4f (az = %.1e m/s²)\n\n', u_h, f_h);


%% 2. Quelle répartition de δ fait tourner le drone ?
REPARTITIONS = [1 -1  1 -1;      % moteurs 1, 3 contre 2, 4
                1  1 -1 -1;      % moteurs 1, 2 contre 3, 4
                1 -1 -1  1];     % moteurs 1, 4 contre 2, 3
T_TEST = 4;  T_DEBUT = 1;        % δ appliqué de 1 s à 4 s
n = round(T_TEST / Ts) + 1;
t = (0:n - 1)' * Ts;
marche = double(t >= T_DEBUT);
Y_ref = simuler(HARNESS, u_h * ones(n, 4), Ts, 0);            % sans δ, pour le couplage avec az
fprintf('2. delta = %.2f de %g s à %g s, sans bruit :\n', DELTA_TEST, T_DEBUT, T_TEST);
fprintf('   %-14s %10s %10s %10s %10s %10s %12s\n', 'moteurs', 'wz fin', 'max|wx|', 'max|wy|', ...
        'max|phi|', 'max|theta|', 'Δaz moyen');
score = zeros(1, 3);
Y_test = cell(1, 3);
for r = 1:3
    U = u_h + DELTA_TEST * marche * REPARTITIONS(r, :);
    Y = simuler(HARNESS, U, Ts, 0);
    Y_test{r} = Y;
    wz_fin = Y(end, c('wz'));
    bascule = max(abs(Y(:, c('wx')))) + max(abs(Y(:, c('wy'))));
    score(r) = abs(wz_fin) / (bascule + 1e-9);
    paires = sprintf('%d%d / %d%d', find(REPARTITIONS(r, :) > 0), find(REPARTITIONS(r, :) < 0));
    fprintf('   %-14s %10.4f %10.4f %10.4f %10.4f %10.4f %12.2e\n', paires, wz_fin, ...
            max(abs(Y(:, c('wx')))), max(abs(Y(:, c('wy')))), max(abs(Y(:, c('phi')))), ...
            max(abs(Y(:, c('theta')))), mean(Y(t >= T_DEBUT, c('az')) - Y_ref(t >= T_DEBUT, c('az'))));
end
[~, r_lacet] = max(score);
wz_fin = Y_test{r_lacet}(end, c('wz'));
if abs(wz_fin) < 1e-6
    error('Aucune répartition ne fait tourner le drone : le simulateur ne produit pas de lacet.');
end
S = REPARTITIONS(r_lacet, :) * sign(wz_fin);   % δ > 0  ->  wz > 0
fprintf('   -> lacet : moteurs u_i = u + s_i·δ avec s = [%s]\n\n', num2str(S));

figure('Name', 'Test des répartitions', 'Position', [100 100 900 600]);
grandeurs = {'wz', 'wx', 'wy', 'psi'};
for g = 1:4
    subplot(2, 2, g); hold on; grid on;
    for r = 1:3, plot(t, Y_test{r}(:, c(grandeurs{g}))); end
    title(grandeurs{g}); xlabel('t [s]');
end
legend('13 / 24', '12 / 34', '14 / 23', 'Location', 'best');
sgtitle(sprintf('\\delta = %.2f à partir de t = %g s, u = u_h', DELTA_TEST, T_DEBUT));
exportgraphics(gcf, fullfile('figures', 'test_repartition.png'));

% Balayages avec la répartition retenue (sans bruit) : sens de rotation et couplage avec l'altitude
%  - δ variable (dont δ < 0) en vol stationnaire
%  - u variable à δ = DELTA_TEST : l'efficacité de δ dépend de la pente de tanh
DELTAS = [-0.05 -0.02 0.01 0.02 0.05 0.1 0.2];
U_NIVEAUX = [0 0.2 u_h 0.6 0.8];
Y_delta = zeros(numel(DELTAS), n, numel(NOMS));
for i = 1:numel(DELTAS)
    Y_delta(i, :, :) = simuler(HARNESS, u_h + DELTAS(i) * marche * S, Ts, 0);
end
Y_u = zeros(numel(U_NIVEAUX), n, numel(NOMS));
Y_u_ref = zeros(numel(U_NIVEAUX), n, numel(NOMS));
for i = 1:numel(U_NIVEAUX)
    Y_u(i, :, :) = simuler(HARNESS, U_NIVEAUX(i) + DELTA_TEST * marche * S, Ts, 0);
    Y_u_ref(i, :, :) = simuler(HARNESS, U_NIVEAUX(i) * ones(n, 4), Ts, 0);
end
t_test = t;
Y_rep = permute(cat(3, Y_test{:}), [3 1 2]);                   % 3 x n x 18
save('tests_lacet.mat', 'Ts', 'NOMS', 'u_h', 't_test', 'T_DEBUT', 'DELTA_TEST', 'REPARTITIONS', ...
     'S', 'Y_ref', 'Y_rep', 'DELTAS', 'Y_delta', 'U_NIVEAUX', 'Y_u', 'Y_u_ref');
fprintf('   Écrit : tests_lacet.mat (tests des répartitions et balayages en δ et en u)\n\n');


%% 3. Données d'identification : u et δ en paliers aléatoires
n = round(T_VOL / Ts) + 1;
t = (0:n - 1)' * Ts;
u = zeros(N_VOLS, n);  delta = zeros(N_VOLS, n);  moteurs = zeros(N_VOLS, n, 4);
entrees = Simulink.SimulationInput.empty;
for v = 1:N_VOLS
    u(v, :) = paliers(n, -1, 1);                                 % même plage que les données fournies
    delta(v, :) = paliers_avec_zeros(n, DELTA_MAX);
    M = min(max(u(v, :)' + delta(v, :)' * S, -1), 1);           % commandes moteur bornées à [-1, 1]
    moteurs(v, :, :) = M;
    entrees(v) = entree(HARNESS, M, Ts, G_NOISE);
end
fprintf('3. Simulation de %d vols de %g s (g_noise = %g)...\n', N_VOLS, T_VOL, G_NOISE);
sorties = sim(entrees, 'ShowProgress', 'off', 'StopOnError', 'on');

donnees = struct();
for k = 1:numel(NOMS), donnees.(NOMS{k}) = zeros(N_VOLS, n); end
for v = 1:N_VOLS
    Y = sorties(v).Y_out;
    for k = 1:numel(NOMS), donnees.(NOMS{k})(v, :) = Y(:, k)'; end
end
close_system(HARNESS, 0);

% donnees_lacet.mat : une ligne = un vol, une colonne = un pas (t = 0, Ts, ..., T_VOL)
% la mesure du pas k dépend des commandes des pas 0 à k-1 (pas de transmission directe)
psi = donnees.psi;  wz = donnees.wz;  w_dot_z = donnees.w_dot_z;  az = donnees.az;
z = donnees.z;  phi = donnees.phi;  theta = donnees.theta;  wx = donnees.wx;  wy = donnees.wy;
g_noise = G_NOISE;  repartition = S;
save('donnees_lacet.mat', 't', 'Ts', 'g_noise', 'u_h', 'repartition', 'u', 'delta', 'moteurs', ...
     'psi', 'wz', 'w_dot_z', 'az', 'z', 'phi', 'theta', 'wx', 'wy');
fprintf('   Écrit : donnees_lacet.mat (%d vols x %d pas)\n', N_VOLS, n);
fprintf('   wz dans [%.2f, %.2f] rad/s, psi dans [%.1f, %.1f] rad, max|wx|, |wy| = %.2e, %.2e rad/s\n', ...
        min(wz(:)), max(wz(:)), min(psi(:)), max(psi(:)), max(abs(wx(:))), max(abs(wy(:))));

figure('Name', 'Exemple de vol', 'Position', [100 100 900 700]);
subplot(4, 1, 1); stairs(t, [u(1, :)', delta(1, :)']); grid on; legend('u', '\delta'); ylabel('commande');
subplot(4, 1, 2); plot(t, w_dot_z(1, :)); grid on; ylabel('w\_dot\_z [rad/s²]');
subplot(4, 1, 3); plot(t, wz(1, :)); grid on; ylabel('wz [rad/s]');
subplot(4, 1, 4); plot(t, psi(1, :)); grid on; ylabel('psi [rad]'); xlabel('t [s]');
exportgraphics(gcf, fullfile('figures', 'exemple_vol.png'));


%% Fonctions locales

function construire_harness(nom)
% UAV_model (sous-système référencé) entre une source U_in (4 commandes moteur, bloquées sur Ts)
% et un To Workspace Y_out (les 18 sorties, échantillonnées à Ts).
if bdIsLoaded(nom), close_system(nom, 0); end
new_system(nom);
add_block('simulink/Sources/From Workspace', [nom '/U'], 'VariableName', 'U_in', ...
          'SampleTime', 'Ts', 'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value');
add_block('simulink/Signal Routing/Demux', [nom '/Demux'], 'Outputs', '4');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [nom '/UAV'], ...
          'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Signal Routing/Mux', [nom '/Mux'], 'Inputs', '18');
add_block('simulink/Sinks/To Workspace', [nom '/Y'], 'VariableName', 'Y_out', ...
          'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Sinks/Terminator', [nom '/T']);
add_line(nom, 'U/1', 'Demux/1');
for i = 1:4,  add_line(nom, sprintf('Demux/%d', i), sprintf('UAV/%d', i)); end
for i = 1:18, add_line(nom, sprintf('UAV/%d', i), sprintf('Mux/%d', i)); end
add_line(nom, 'UAV/19', 'T/1');
add_line(nom, 'Mux/1', 'Y/1');
end

function in = entree(nom, U, Ts, g_noise)
% Une simulation : U (n x 4) = commandes des pas 0..n-1, nouvelles graines de bruit.
n = size(U, 1);
t = (0:n - 1)' * Ts;
in = Simulink.SimulationInput(nom);
in = in.setVariable('U_in', [t, U]);
in = in.setVariable('Ts', Ts);
in = in.setVariable('g_noise', g_noise);
in = in.setModelParameter('StopTime', num2str(t(end)));
graines = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'phi', 'psi', 'theta', ...
           'wx', 'wy', 'wz', 'w_dot_x', 'w_dot_y', 'w_dot_z'};
for i = 1:numel(graines)
    in = in.setVariable(['seed_' graines{i}], randi(10000));
end
end

function Y = simuler(nom, U, Ts, g_noise, t_min, t_max, colonne)
% Simule et renvoie les sorties (n x 18), ou la colonne demandée entre t_min et t_max.
out = sim(entree(nom, U, Ts, g_noise));
Y = out.Y_out;
if nargin > 4
    t = (0:size(Y, 1) - 1)' * Ts;
    Y = Y(t >= t_min & t <= t_max, colonne);
end
end

function s = paliers(n, bas, haut)
% Signal en paliers : valeurs uniformes dans [bas, haut], tenues 4 à 20 pas (0.2 à 1 s).
s = zeros(1, n);
k = 1;
while k <= n
    d = randi([4 20]);
    s(k:min(n, k + d - 1)) = bas + (haut - bas) * rand;
    k = k + d;
end
end

function s = paliers_avec_zeros(n, amplitude)
% Paliers dans [-amplitude, amplitude], dont un sur trois est nul : le modèle doit aussi voir δ = 0.
s = paliers(n, -amplitude, amplitude);
debut = [true, diff(s) ~= 0];               % premier pas de chaque palier
numero = cumsum(debut);                     % numéro du palier de chaque pas
nul = rand(1, numero(end)) < 1/3;
s(nul(numero)) = 0;
end
