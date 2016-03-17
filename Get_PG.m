function   [hPx,  lPx] =  Get_PG( hiresIm, loresIm, gmmConf)

win=gmmConf.win;
ps=gmmConf.ps;
nlsp=gmmConf.nlsp;
step=gmmConf.step;
delta=gmmConf.delta;
[h, w]  =  size(hiresIm);
S         =  win;
maxr         =  h-ps+1;
maxc         =  w-ps+1;
r         =  [1:step:maxr];
r         =  [r r(end)+1:maxr];
c         =  [1:step:maxc];
c         =  [c c(end)+1:maxc];
hX_temp = zeros(ps^2,maxr*maxc,'single');
lX_temp= zeros(4* ps^2,maxr*maxc,'single');

lX1= zeros(ps^2,maxr*maxc,'single');
lX2= zeros(ps^2,maxr*maxc,'single');
lX3= zeros(ps^2,maxr*maxc,'single');
lX4= zeros(ps^2,maxr*maxc,'single');

loresIm1=loresIm(:,:,1);
loresIm2=loresIm(:,:,2);
loresIm3=loresIm(:,:,3);
loresIm4=loresIm(:,:,4);

 hPx = [];     lPx = []; 

if nlsp ==1
    k    =  0;
    for i  = 1:ps
        for j  = 1:ps
            k    =  k+1;
            blk_h     =  hiresIm(r-1+i,c-1+j);
            hPx(k,:) =  blk_h(:)';

            blk_l1     =  loresIm1(r-1+i,c-1+j);
            blk_l2     =  loresIm2(r-1+i,c-1+j);
            blk_l3     =  loresIm3(r-1+i,c-1+j);
            blk_l4     =  loresIm4(r-1+i,c-1+j);   
           
            lPx1(k,:) = blk_l1(:)';
            lPx2(k,:) = blk_l2(:)';
            lPx3(k,:) = blk_l3(:)';
            lPx4(k,:) = blk_l4(:)';
            
        end
    end
          lPx =  [lPx1; lPx2; lPx3; lPx4];
          clear lPx1 lPx2 lPx3 lPx4;
%           [lPx, lV_pca]=dataPCA(lPx);
    
else     %  nlsp !=1
    k    =  0;
    for i  = 1:ps
        for j  = 1:ps
            k    =  k+1;
            blk_h  =  hiresIm(i:end-ps+i,j:end-ps+j);
            hX_temp(k,:) =  blk_h(:)';
        end
    end
    
     k    =  0;
    for i  = 1:ps
        for j  = 1:ps
            k    =  k+1;           
           blk_l1  =  loresIm1(i:end-ps+i,j:end-ps+j);
           blk_l2  =  loresIm2(i:end-ps+i,j:end-ps+j);
           blk_l3  =  loresIm3(i:end-ps+i,j:end-ps+j);
           blk_l4  =  loresIm4(i:end-ps+i,j:end-ps+j);         
           lX1(k,:) =  blk_l1(:)';
           lX2(k,:) =  blk_l2(:)';
           lX3(k,:) =  blk_l3(:)';
           lX4(k,:) =  blk_l4(:)';        
        end
    end      
       lX_temp =  [lX1; lX2; lX3; lX4];
%        [ lX, lV_pca]= dataPCA (lX);
    
    % Index image
    Index     =   (1:maxr*maxc);
    Index    =   reshape(Index, maxr, maxc);
    N1    =   length(r);
    M1    =   length(c);
    blk_arr   =  zeros(nlsp, N1*M1 );
    for  i  =  1 :N1
        for  j  =  1 : M1
            row     =   r(i);
            col     =   c(j);
            off     =  (col-1)*maxr + row;
            off1    =  (j-1)*N1 + i;
            
            rmin    =   max( row-S, 1 );
            rmax    =   min( row+S, maxr );
            cmin    =   max( col-S, 1 );
            cmax    =   min( col+S, maxc );
            
            idx     =   Index(rmin:rmax, cmin:cmax);
            idx     =   idx(:);
            neighbor       =   hX_temp(:,idx);
            seed       =   hX_temp(:,off);
            
            dis     =   (neighbor(1,:) - seed(1)).^2;
            for k = 2:ps^2
                dis   =  dis + (neighbor(k,:) - seed(k)).^2;
            end
            dis = dis./ps^2;
            [~,ind]   =  sort(dis);
            indc        =  idx( ind( 1 : nlsp ) );
            blk_arr(:,off1)  =  indc;
            
            hX_nl = hX_temp(:,indc); % or X_nl = neighbor(:,ind( 1 : nlsp ));
            % Removes DC component from image patch group
            hDC = mean(hX_nl,2);
            hX_nl = bsxfun(@minus, hX_nl, hDC);
            
            lX_nl =lX_temp(:,indc); % or X_nl = neighbor(:,ind( 1 : nlsp ));
            % Removes DC component from image patch group
            lDC = mean(lX_nl,2);
            lX_nl = bsxfun(@minus, lX_nl, lDC);
            
            % Select the smooth patches
%             sv_h=var(hX_nl);
%             sv_l=var(lX_nl);
            
%             if max(sv_h)*100 <= delta(1) && max(sv_l)*100 <= delta(2)
%                 hPx0 = [hPx0 hX_nl];
%                 lPx0 = [lPx0 lX_nl];
%                 lX_nl=[]; hX_nl=[];
%             else
                hPx = [hPx hX_nl];
                lPx =  [lPx lX_nl];
                lX_nl=[];  hX_nl=[];
%             end
        end
    end
end
clear hX_temp  lX_temp lX1 lX2 lX3 lX4;
clear hiresIm loresIm1 loresIm2 loresIm3 loresIm4;
end
