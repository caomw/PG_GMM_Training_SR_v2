close all; clear all; clc;
%%  Set the parameters
[gmmConf, dataConf] = ParaSet;

%% load high and low resolution image data
% TD_path      =   '.\TrainingData';
% [hiresImgs, loresImgs]= prepareImg( load_images(glob( TD_path, '*.bmp')), dataConf );
prepareImg=load('prepareImgs.mat');
hiresImgs=prepareImg.hiresImgs;
loresImgs=prepareImg.loresImgs;
im_num    =   size (hiresImgs);

hX     =  [];    lX     =  [];   

%% get group patches for high resolution images
for  i  =  1: im_num
    imhires = hiresImgs{i};   
    imlores =loresImgs{i};
    [hPx, lPx] =  Get_PG( imhires, imlores, gmmConf);    
    sprintf('Get PG for the %d / %d images...\n', i, im_num);
    clear imhires imlores;    
    hX   = [hX hPx];
    lX   = [lX lPx];
    clear hPx  lPx  ;
end

% num_lX=size(lX,2);   
[lX, lV_pca] = dataPCA(lX);
dataConf. lV_pca=lV_pca;

name_mat = sprintf('Get_PG.mat');
save(name_mat,'hX','lX','lV_pca');

%% PG-GMM Training for hires Imgs
sprintf('Start PG training for hires Imgs...\n');
[model_h, llh_h, cls_idh] = emgm(hX, gmmConf);     %cls_idx :  label vector for all X
sprintf('Trained PG for hires Imgs with %d components in total...\n', max(cls_idh(:)));

[s_idh, seg_h]    =  Proc_cls_idx( cls_idh );        % seg(i):    第（i-1）个 高斯成分中含有 x 数据的个数，
                                                                   %s_idx(1 : seg(2)）: 属于第 1 个高斯成分的 所有x 数据的 id
% gmmConf.cls_num = size(model_h.R,2)+1;
% cls_num=gmmConf.cls_num;
%  model_h.R = []; % R is now useless
% model_h.means(:,cls_num) = mean(hX0,2);
% model_h.covs(:,:,cls_num) = cov(hX0');
% length0 = size(hX0,2)/nlsp;
% model_h.mixweights = [model_h.mixweights length0/(length0 + length(cls_idh))]/(sum(model_h.mixweights) + length0/(length0 + length(cls_idh)));
% model_h.nmodels = model_h.nmodels + 1;
clear hX ;
ps=gmmConf.ps;
cls_num=gmmConf.cls_num;
% Get GMM dictionaries and regularization parameters
GMM_DH    =  zeros(ps^4, cls_num);
GMM_SH    =  zeros(ps^2, cls_num);
for  i  =  1 : length(seg_h)-1
    idx    =   s_idh(seg_h(i)+1:seg_h(i+1));          % 属于第 i 个高斯成分的所有x的id
    cls    =   cls_idh(idx(1));
    [P,S,~] = svd(model_h.covs(:,:,i));
    S = diag(S);
    GMM_DH(:,cls)    =  P(:);
    GMM_SH(:,cls)    =  S;
end
% [P0,S0,~] = svd(model_h.covs(:,:,cls_num));
% S0 = diag(S0);
% GMM_DH(:,cls_num)    =  P0(:);
% GMM_SH(:,cls_num)    =  S0;
clear P S ;

%% PG-GMM Training for lores Imgs
sprintf('Start PG training for lores Imgs...\n');
[model_l, llh_l, cls_idl] = emgm(lX, gmmConf);     %cls_idx :  label vector for all X
sprintf('Trained PG for lores Imgs with %d components in total...\n', max(cls_idl(:)));

[s_idl, seg_l]    =  Proc_cls_idx( cls_idl );        % seg(i):    第（i-1）个 高斯成分中含有 x 数据的个数，
                                                                   %s_idx(1 : seg(2)）: 属于第 1 个高斯成分的 所有x 数据的 id
% gmmConf.cls_num = size(model_l.R,2)+1;
% cls_num=gmmConf.cls_num;
% model_l.R = []; % R is now useless
% model_l.means(:,cls_num) = mean(lX0,2);
% model_l.covs(:,:,cls_num) = cov(lX0');
% length0 = size(lX0,2)/nlsp;
% model_l.mixweights = [model_l.mixweights length0/(length0 + length(cls_idl))]/(sum(model_l.mixweights) + length0/(length0 + length(cls_idl)));
% model_l.nmodels = model_l.nmodels + 1;
dlX=size(lX,1);
clear lX ;
% Get GMM dictionaries and regularization parameters
GMM_DL    =  zeros(dlX^2, cls_num);
GMM_SL    =  zeros(dlX, cls_num);
for  i  =  1 : length(seg_l)-1
    idx    =   s_idl(seg_l(i)+1:seg_l(i+1));          % 属于第 i 个高斯成分的所有x的id
    cls    =   cls_idl(idx(1));
    [P,S,~] = svd(model_l.covs(:,:,i));
    S = diag(S);
    GMM_DL(:,cls)    =  P(:);
    GMM_SL(:,cls)    =  S;
end
% [P0,S0,~] = svd(model_l.covs(:,:,cls_num));
% S0 = diag(S0);
% GMM_DL(:,cls_num)    =  P0(:);
% GMM_SL(:,cls_num)    =  S0;
clear P S ;

% delta=gmmConf.delta;
%% save PG-GMM model 
% name_h = sprintf('Hires_PG_GMM_%dx%d_win%d_nlsp%d_delta_h%2.3f_cls%d.mat',ps,ps,win,nlsp,delta,cls_num);
% save(name_h,'model','nlsp','GMM_DH','GMM_SH','cls_num','delta(1)','ps','win');
% 
% name_l = sprintf('Lores_PG_GMM_%dx%d_win%d_nlsp%d_delta_l%2.3f_cls%d.mat',ps,ps,win,nlsp,delta,cls_num);
% save(name_l,'model','nlsp','GMM_DL','GMM_SL','cls_num','delta(2)','ps','win','lV_pca');

name_h = sprintf('Hires_PG_GMM_%dx%d_win%2.3f_cls%d.mat',ps,ps,cls_num);
save(name_h,'model_h','GMM_DH','GMM_SH','gmmConf','dataConf');

name_l = sprintf('Lores_PG_GMM_%dx%d_win%2.3f_cls%d.mat',ps,ps,cls_num);
save(name_l,'model_l','GMM_DL','GMM_SL','lV_pca','gmmConf','dataConf');