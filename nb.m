function [M_res] = nb( train_data, train_target, test_data, test_target )
    
    % Guardamos los largos de las Matrices de entrada
    % -> num_class      : n�mero de clases
    % -> num_train_docs : n�mero de documentos de entrenamiento
    % -> num_test_docs  : n�mero de documentos de entrenamiento
    % -> num_feat       : n�mero de features
    
    [num_class, num_train_docs] = size(train_target);
    [num_train_docs, num_feat] = size(train_data);
    [num_test_docs, num_feat] = size(test_data);
   
    % Calculamos P(t_i|c) y P(t_i|�c)
    % -> c = 1 : c
    % -> nc= 0 : si el doc pertenece a �c
    
    c = 1;
    nc = 0;
    M_dinc = zeros(num_class, 1);
    M_dinnc = zeros(num_class, 1);
    
    
    % Contamos los documentos en una clase c y �c
    for class_i=1:num_class
        M_dinc(class_i, 1) = sum(train_target(class_i,:) == c);
        M_dinnc(class_i, 1) = sum(train_target(class_i,:)== nc);
    end
    
    % Documentos que est�n presentes en c_i y cotienen feat_i
    M_dincwithf = zeros(num_class, num_feat);
    % Documentos que est�n presentes en �c_i y contienen feat_i
    M_dinncwithf = zeros(num_class, num_feat);
    
    for feat_i=1:num_feat
        for class_i=1:num_class
            for doc_i=1:num_train_docs
                if (train_target(class_i, doc_i) == c && train_data(doc_i, feat_i) > 0)
                    M_dincwithf(class_i, feat_i) = M_dincwithf(class_i, feat_i) + 1;
                end
                if (train_target(class_i, doc_i) == nc && train_data(doc_i, feat_i) > 0)
                    M_dinncwithf(class_i, feat_i) = M_dinncwithf(class_i, feat_i) + 1;
                end
            end
        end
    end
    
    % Prob de P(c_i)
    M_pc = zeros(num_class, 2);
    % Prob de P(�c_i)
    M_pnc = zeros(num_class, 2);
    
    for class_i=1:num_class
        M_pc(class_i,1) = (M_dinc(class_i, 1)+1)/(num_train_docs + num_class);
        M_pc(class_i,2) = log10(M_pc(class_i, 1));
        M_pnc(class_i,1) = (M_dinnc(class_i, 1) + 1)/(num_train_docs + num_class);
        M_pnc(class_i,2) = log10(M_pnc(class_i, 1));
        
        %M_pc(class_i,1) = M_dinc(class_i, 1)/num_train_docs;
        %M_pc(class_i,2) = max(log10(M_pc(class_i, 1)), -100);
        %M_pnc(class_i,1) = M_dinnc(class_i, 1)/num_train_docs;
        %M_pnc(class_i,2) = max(log10(M_pnc(class_i, 1)), -100);
    end
    
    % Guardamos la P(t_i|c) y P(t_i|�c)
    M_ptc = zeros(num_class, num_feat);
    M_ptnc = zeros(num_class, num_feat);
    
    for class_i=1:num_class
        for feat_i=1:num_feat
            M_ptc(class_i, feat_i) = (M_dincwithf(class_i, feat_i) + 1)/(num_feat + sum(M_dincwithf(class_i, :)));
            M_ptnc(class_i, feat_i) = (M_dinncwithf(class_i, feat_i) + 1)/(num_feat + sum(M_dinncwithf(class_i, :)));
        end
    end
    
    % Creamos la matriz de correlaci�n
    M_corr = zeros(num_class, num_class);
    curr_docs_in_cj = 0;
    for class_i=1:num_class
        for class_j=1:num_class
            for doc_i=1:num_train_docs
                if(train_target(class_i, doc_i) == 1)
                    if(train_target(class_j, doc_i) == 1)
                        curr_docs_in_cj = curr_docs_in_cj + 1;
                    end
                end
            end
            M_corr(class_i, class_j) = curr_docs_in_cj/M_dinc(class_i, 1);
            if isnan(M_corr(class_i, class_j))
                M_corr(class_i, class_j) = 0;
            end
            curr_docs_in_cj = 0;
        end
    end
    
    % Contamos y sumamos los n�mero distintos de 0 y 1 para calcular un
    % threshold
    % corr_count = sum(sum(M_corr~=0&M_corr~=1));
    % corr_sum = sum(M_corr(M_corr~=1&M_corr~=0));
    corr_thres = 0.4;
    
    M_corr2 = M_corr;
    
    M_corr2(M_corr2 >= corr_thres) = 1;
    M_corr2(M_corr2 < corr_thres) = 0;
    
    % TESTING %
    M_fdc = zeros(num_class, num_test_docs);
    M_fdnc = zeros(num_class, num_test_docs);
    M_prob = zeros(num_class, num_test_docs);
    
    for doc_i=1:num_test_docs
        for class_i=1:num_class
            fdc = 0;
            fdnc = 0;
            for feat_i=1:num_feat
                w = test_data(doc_i, feat_i);
                ptc = M_ptc(class_i, feat_i);
                ptnc = M_ptnc(class_i, feat_i);
                %fdc = fdc + w*log10((ptc)/(1-ptc)) + log10(1-ptc); 
                %fdnc = fdnc + w*log10((ptnc)/(1-ptnc)) + log10(1-ptnc); 
                fdc = fdc + w*log10(((ptc)*(1-ptnc))/((1-ptc)*ptnc));
            end
                %M_fdc(class_i, doc_i)  = (fdc + M_pc(class_i, 2));
                M_fdc(class_i, doc_i)  = fdc;
                %M_fdnc(class_i, doc_i) = fdnc + M_pnc(class_i, 2);
                M_prob(class_i, doc_i) = M_fdc(class_i, doc_i) - M_fdnc(class_i, doc_i);
        end
    end
    
    % Calculamos el Umbral para cada documento
    % Est� dado por el m�ximo de la muestra - la desviaci�n est�ndar
    M_threshold = zeros(num_test_docs, 1);
    for doc_i=1:num_test_docs
        M_threshold(doc_i, 1) = max(M_fdc(:, doc_i)) - 0.2*std2(M_fdc(:, doc_i));
        %M_threshold(doc_i, 1) = sum(M_fdc(:, doc_i))/num_class;
    end
    
    % Aplicamos un umbral a la decision final
    M_res = zeros(num_class, num_test_docs);
    for class_i=1:num_class
        for doc_i=1:num_test_docs
            if(M_fdc(class_i, doc_i) > M_threshold(doc_i, 1))
            %if(M_fdc(class_i, doc_i) > M_fdnc(class_i, doc_i))
                M_res(class_i, doc_i) = 1;
            end
        end
    end
    
    M_res2 = M_res;
    
    % SEGUNDA FASE
    for class_i=1:num_class
        for class_j=1:num_class
            for doc_i=1:num_test_docs
                if(M_res2(class_i, doc_i) == 1 && M_corr2(class_i, class_j) == 1)
                    M_res(class_j, doc_i) = 1;
                end
            end
        end
    end
    
    get_measures(M_res, test_target);
    
end


