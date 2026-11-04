%% Génération des bases de données du tangage avec UAV_model.slx
%  Entrée : u_theta, commande différentielle de tangage. Le drone est en « + » :
%           moteur avant u1 = u0 - u_theta, moteur arrière u3 = u0 + u_theta, u2 = u4 = u0 + u_psi
%  Sorties : ax (accélération sur x) et a_theta (w_dot_y, accélération angulaire de tangage)
%
%  Trois contraintes physiques :
%  - poussée collective u0 à l'équilibre (az = 0 à plat) : en chute libre (u0 = 0) il n'y a pas de poussée à
%    incliner, donc ax ~ 0 et la base sur ax serait inutile ;
%  - le tangage est un double intégrateur rapide (environ 170 rad/s² par unité de u_theta) : en boucle ouverte,
%    le drone se retourne en moins d'une seconde. Un PD stabilise theta pendant la génération et une excitation
%    aléatoire s'y ajoute (paliers comme dans la base fournie). On enregistre le u_theta réellement appliqué.
%    Le PD utilise theta estimé en intégrant le gyromètre wy : avec g_noise = 0.01, la sortie theta du
%    simulateur a un bruit d'écart-type 3,5 rad (inutilisable), contre 0,0005 pour wy, ax et a_theta.
%  - les couples de lacet des moteurs 1 et 3 ne se compensent qu'au premier ordre : sans correction, le lacet
%    dérive (-105° en 100 s) et l'accélération passe de x vers y. Un second PD tient le lacet à 0 avec les
%    moteurs 2 et 4 (u_psi = Kpsi.psi_est + Kdpsi.wz, psi estimé en intégrant le gyromètre wz).
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', 'Matlab files'));
% Compilation de Simulink dans un dossier au chemin court (limite de 260 caractères de Windows)
Simulink.fileGenControl('set', 'CacheFolder', fullfile(tempdir, 'sluav'), ...
                        'CodeGenFolder', fullfile(tempdir, 'sluav'), 'createDir', true);

%% 1. Paramètres
N_ESSAIS = 3000;          % nombre d'essais (lignes des fichiers), comme la base fournie
N = 2000;                 % pas par essai : 100 s à Ts = 0.05 s
N_REPOS = 40;             % pas au repos au début de chaque essai
random_init;              % graines du bruit (écrase Ts et g_noise)
Ts = 0.05;
g_noise = 0.01;           % bruit des capteurs prévu par random_init pour la génération de données
rng(2026);                % excitation reproductible
u0 = 0.386;               % poussée collective d'équilibre (az = 0 à plat), identifiée sur le RNN de az
Kp = 0.02;                % PD de stabilisation : u_theta = exc - Kp.theta_est - Kd.wy
Kd = 0.015;
A_EXC = 0.007;            % amplitude de l'excitation : theta d'environ ±A_EXC/Kp = ±0,35 rad (20°)
Kpsi = 0.04;              % PD de lacet (moteurs 2 et 4) : u_psi = Kpsi.psi_est + Kdpsi.wz
Kdpsi = 0.03;

%% 2. Modèle de génération : excitation + PD -> mélange des moteurs -> UAV_model
h = 'boucle_tangage';
construire_modele(h);
t = (0:N-1)' * Ts;

%% 3. Essais (fast restart : le modèle n'est compilé qu'une fois)
U_THETA = zeros(N_ESSAIS, N); AX = U_THETA; A_THETA = U_THETA; THETA_MAX = zeros(N_ESSAIS, 1); PSI_MAX = THETA_MAX;
set_param(h, 'FastRestart', 'on');
t0 = tic;
for i = 1:N_ESSAIS
    exc = zeros(N, 1);                                  % paliers aléatoires après le repos
    k = N_REPOS + 1;
    while k <= N
        L = randi([5 39]);
        exc(k:min(k+L-1, N)) = A_EXC * (2*rand - 1);
        k = k + L;
    end
    EXC = [t exc];
    tirer_graines;                                      % bruit différent à chaque essai
    out = sim(h, 'StopTime', num2str((N-1)*Ts));
    U_THETA(i, :) = out.utheta(:)';
    AX(i, :) = out.ax(:)';
    A_THETA(i, :) = out.atheta(:)';
    THETA_MAX(i) = max(abs(out.theta_est(:)));
    PSI_MAX(i) = max(abs(out.psi_est(:)));
    if i == 1                                           % essai d'exemple pour la figure
        ex = struct('t', t, 'exc', exc, 'utheta', out.utheta(:), 'theta', out.theta_est(:), ...
                    'ax', out.ax(:), 'atheta', out.atheta(:), 'az', out.az(:));
    end
    if mod(i, 250) == 0
        fprintf('%4d / %d essais (%.0f s)\n', i, N_ESSAIS, toc(t0));
    end
end
set_param(h, 'FastRestart', 'off');
save_system(h);

%% 4. Vérifications
fprintf('\nu_theta : [%+.4f ; %+.4f]   ax : [%+.2f ; %+.2f] m/s²   a_theta : [%+.1f ; %+.1f] rad/s²\n', ...
        min(U_THETA(:)), max(U_THETA(:)), min(AX(:)), max(AX(:)), min(A_THETA(:)), max(A_THETA(:)));
fprintf('theta estimé : max %.2f rad (essais au-delà de 0,6 rad : %d)\n', max(THETA_MAX), sum(THETA_MAX > 0.6));
fprintf('lacet estimé : max %.3f rad\n', max(PSI_MAX));
fprintf('NaN : %d\n', sum(isnan([U_THETA(:); AX(:); A_THETA(:)])));

%% 5. Écriture (1 ligne = 1 essai, 1 colonne = 1 pas, comme bdd_in_mat_05 / bdd_out_mat_05), compressée
ecrire_csv_gz('bdd_in_utheta.csv', U_THETA);
ecrire_csv_gz('bdd_out_ax.csv', AX);
ecrire_csv_gz('bdd_out_atheta.csv', A_THETA);

% Figure : un essai
figure('Name', 'Essai exemple', 'Position', [100 100 900 800]);
subplot(4, 1, 1); stairs(ex.t, ex.utheta); hold on; stairs(ex.t, ex.exc, '--'); grid on;
ylabel('u_\theta'); legend('appliqué (PD + excitation)', 'excitation'); title('Essai 1 : boucle de tangage');
subplot(4, 1, 2); plot(ex.t, ex.theta); grid on; ylabel('\theta estimé [rad]');
subplot(4, 1, 3); plot(ex.t, ex.atheta); grid on; ylabel('a_\theta [rad/s²]');
subplot(4, 1, 4); plot(ex.t, ex.ax); grid on; ylabel('a_x [m/s²]'); xlabel('t [s]');
exportgraphics(gcf, 'essai_exemple.png');
fprintf('Bases écrites : bdd_in_utheta.csv.gz, bdd_out_ax.csv.gz, bdd_out_atheta.csv.gz (%d essais x %d pas)\n', N_ESSAIS, N);


%% Fonctions locales
function construire_modele(h)
% Excitation EXC + PD (theta estimé par intégration de wy) -> u_theta -> moteurs 1 et 3 -> UAV_model
if bdIsLoaded(h), close_system(h, 0); end
if isfile([h '.slx']), delete([h '.slx']); end
new_system(h);
load_system('EINT_system_simulation');                 % même solveur que l'exemple fourni
cs = copy(getActiveConfigSet('EINT_system_simulation')); cs.Name = 'config_EINT';
attachConfigSet(h, cs, true); setActiveConfigSet(h, 'config_EINT');
add_block('simulink/Sources/From Workspace', [h '/Excitation'], 'VariableName', 'EXC', 'Interpolate', 'off', ...
          'OutputAfterFinalValue', 'Holding final value', 'SampleTime', 'Ts');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [h '/UAV_model'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Discrete/Discrete-Time Integrator', [h '/theta_est'], 'gainval', '1', 'SampleTime', 'Ts');
add_block('simulink/Math Operations/Gain', [h '/Kp'], 'Gain', 'Kp');
add_block('simulink/Math Operations/Gain', [h '/Kd'], 'Gain', 'Kd');
add_block('simulink/Math Operations/Sum', [h '/u_theta'], 'Inputs', '+--');
add_block('simulink/Sources/Constant', [h '/u0'], 'Value', 'u0');
add_block('simulink/Math Operations/Sum', [h '/u1 avant'], 'Inputs', '+-');
add_block('simulink/Math Operations/Sum', [h '/u3 arriere'], 'Inputs', '++');
add_line(h, 'Excitation/1', 'u_theta/1');
add_line(h, 'UAV_model/11', 'theta_est/1');            % sortie 11 : wy (gyromètre)
add_line(h, 'theta_est/1', 'Kp/1');      add_line(h, 'Kp/1', 'u_theta/2');
add_line(h, 'UAV_model/11', 'Kd/1');     add_line(h, 'Kd/1', 'u_theta/3');
add_line(h, 'u0/1', 'u1 avant/1');       add_line(h, 'u_theta/1', 'u1 avant/2');
add_line(h, 'u0/1', 'u3 arriere/1');     add_line(h, 'u_theta/1', 'u3 arriere/2');
% Lacet : u2 = u4 = u0 + u_psi, u_psi = Kpsi.psi_est + Kdpsi.wz (psi estimé en intégrant le gyromètre wz)
add_block('simulink/Discrete/Discrete-Time Integrator', [h '/psi_est'], 'gainval', '1', 'SampleTime', 'Ts');
add_block('simulink/Math Operations/Gain', [h '/Kpsi'], 'Gain', 'Kpsi');
add_block('simulink/Math Operations/Gain', [h '/Kdpsi'], 'Gain', 'Kdpsi');
add_block('simulink/Math Operations/Sum', [h '/u_psi'], 'Inputs', '++');
add_block('simulink/Math Operations/Sum', [h '/u2 u4 lacet'], 'Inputs', '++');
add_line(h, 'UAV_model/12', 'psi_est/1');               % sortie 12 : wz (gyromètre)
add_line(h, 'psi_est/1', 'Kpsi/1');      add_line(h, 'Kpsi/1', 'u_psi/1');
add_line(h, 'UAV_model/12', 'Kdpsi/1');  add_line(h, 'Kdpsi/1', 'u_psi/2');
add_line(h, 'u0/1', 'u2 u4 lacet/1');    add_line(h, 'u_psi/1', 'u2 u4 lacet/2');
add_line(h, 'u1 avant/1', 'UAV_model/1'); add_line(h, 'u2 u4 lacet/1', 'UAV_model/2');
add_line(h, 'u3 arriere/1', 'UAV_model/3'); add_line(h, 'u2 u4 lacet/1', 'UAV_model/4');
enregistres = {'utheta', 'u_theta/1'; 'theta_est', 'theta_est/1'; 'psi_est', 'psi_est/1'; ...
               'ax', 'UAV_model/13'; 'az', 'UAV_model/15'; 'atheta', 'UAV_model/17'};
for k = 1:size(enregistres, 1)
    add_block('simulink/Sinks/To Workspace', [h '/enr_' enregistres{k, 1}], 'VariableName', enregistres{k, 1}, ...
              'SaveFormat', 'Array', 'SampleTime', 'Ts');
    add_line(h, enregistres{k, 2}, ['enr_' enregistres{k, 1} '/1']);
end
for k = setdiff(1:19, [11 12 13 15 17])                 % sorties non utilisées
    add_block('simulink/Sinks/Terminator', sprintf('%s/T%d', h, k));
    add_line(h, sprintf('UAV_model/%d', k), sprintf('T%d/1', k));
end
Simulink.BlockDiagram.arrangeSystem(h);
save_system(h);
end

function tirer_graines
% Nouvelles graines pour les 18 bruits de UAV_model (variables du workspace de base, comme random_init)
noms = {'x','y','z','vx','vy','vz','ax','ay','az','phi','psi','theta','wx','wy','wz','w_dot_x','w_dot_y','w_dot_z'};
for n = 1:numel(noms)
    assignin('base', ['seed_' noms{n}], randi(10000));
end
end

function ecrire_csv_gz(nom, M)
% Écrit M en CSV (6 chiffres significatifs, largement sous le bruit) puis le compresse en .csv.gz
fid = fopen(nom, 'w');
fprintf(fid, [repmat('%.6g,', 1, size(M, 2) - 1) '%.6g\n'], M.');
fclose(fid);
gzip(nom);
delete(nom);
end
