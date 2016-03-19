close all; clear all; clc;
%%  Set the parameters
[gmm_conf, data_conf] = ParaSet;

%% load high and low resolution image data
 TD_path      =   '.\Data\TrainingData';
 hires_images= load_images(glob(TD_path, '*.bmp');
 [hires_patches, features_pca, V_pca]= prepare_data(data_conf, hires_images );
 
 data_conf.hires_patches=hires_patches;
 data_conf.features_pca=features_pca;     % lores data set
 data_conf.V_pca=V_pca;
 
%% PG-GMM Training for lores Imgs
% gmm_conf.data = double(features_pca);

[data_dim, data_num ]=size(features_pca);
cls_num=gmm_conf.cls_num;
clear features_pca hires_patches V_pca;

% Training process (will take a while)
tic;
fprintf('Training  GMM with %d components on [%d x %d ] low features \n', ...
    gmm_conf.cls_num, data_dim , data_num);

%  GMM Training for lores patches
[model,llh,cls_idx] = emgm(features_pca, gmm_conf);     %cls_idx :  label vector for all X
[s_idx, seg]    =  Proc_cls_idx( cls_idx );     % seg(i):    第（i-1）个 高斯成分中含有 x 数据的个数，
                                                                   %s_idx(1 : seg(2)）: 属于第 1 个高斯成分的 所有x 数据的 id

% Get GMM dictionaries and regularization parameters
GMM_D    =  zeros(data_dim^4, cls_num);
GMM_S    =  zeros(data_dim^2, cls_num);
for  i  =  1 : length(seg)-1
    idx    =   s_idx(seg(i)+1:seg(i+1));     % 属于第 i 个高斯成分的所有x的id
    cls    =   cls_idx(idx(1));
    [P,S,~] = svd(model.covs(:,:,i));
    S = diag(S);
    GMM_D(:,cls)    =  P(:);
    GMM_S(:,cls)    =  S;
end

name_mat= sprintf('Lores_PG_GMM_dim_%d_cls_%d.mat',data_dim ,cls_num);
save(name_mat,'model_l','GMM_D','GMM_S','data_conf','cls_idx');
