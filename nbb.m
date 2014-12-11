function [Dc, Dnc, Dtc, Dtnc Pc, Pnc, Pcd, Pncd, Pthres, test_predicted] = nbb( train_data, train_target, test_data, test_target )
    
    % Guardamos los largos de las Matrices de entrada
    % -> num_class      : número de clases
    % -> num_train_docs : número de documentos de entrenamiento
    % -> num_test_docs  : número de documentos de entrenamiento
    % -> num_feat       : número de features
    
    [num_class, num_train_docs] = size(train_target);
    [num_train_docs, num_feat] = size(train_data);
    [num_test_docs, num_feat] = size(test_data);
   
    % Calculamos P(t_i|c) y P(t_i|¬c)
    % -> c = 1 : c
    % -> nc= 0 : si el doc pertenece a ¬c
    
    c = 1;
    nc = 0;
    Dc = zeros(num_class, 1);
    Dnc = zeros(num_class, 1);
    
    
    % Contamos los documentos en una clase c y ¬c
    for class_i=1:num_class
        Dc(class_i, 1) = sum(train_target(class_i,:) == c);
        Dnc(class_i, 1) = sum(train_target(class_i,:)== nc);
    end
    
    % Documentos que están presentes en c_i y cotienen feat_i
    Dtc = zeros(num_class, num_feat);
    % Documentos que están presentes en ¬c_i y cotienen feat_i
    Dtnc = zeros(num_class, num_feat);
    for feat_i=1:num_feat
        for class_i=1:num_class
            for doc_i=1:num_train_docs
                if (train_target(class_i, doc_i) == c && train_data(doc_i, feat_i) > 0)
                    Dtc(class_i, feat_i) = Dtc(class_i, feat_i) + 1;
                end
                if (train_target(class_i, doc_i) == nc && train_data(doc_i, feat_i) > 0)
                    Dtnc(class_i, feat_i) = Dtnc(class_i, feat_i) + 1;
                end
            end
        end
    end
    
    % Prob de P(c_i)
    Pc = zeros(num_class, 1);
    Pnc = zeros(num_class, 1);
    
    for class_i=1:num_class
        Pc(class_i,1) = (Dc(class_i, 1)+1)/(2 + num_train_docs);
        Pnc(class_i,1) = (Dnc(class_i, 1)+1)/(2 + num_train_docs);
    end
    
    % Guardamos la P(t_i|c) y P(t_i|¬c)
    Ptc = zeros(num_class, num_feat);
    Ptnc = zeros(num_class, num_feat);
    
    for class_i=1:num_class
        for feat_i=1:num_feat
            Ptc(class_i, feat_i) = (Dtc(class_i, feat_i) + 1)/(2 + Dc(class_i, 1));
            Ptnc(class_i, feat_i) = (Dtnc(class_i, feat_i) + 1)/(2 + Dnc(class_i, 1));
        end
    end
    
    % TESTING
    Pcd = zeros(num_class, num_test_docs);
    Pncd = zeros(num_class, num_test_docs);
    
    for class_i=1:num_class
        probcd = Pc(class_i, 1);
        probncd = Pnc(class_i, 1);
        for doc_i=1:num_test_docs
            prod = 1;
            prodn = 1;
            for feat_i=1:num_feat
                w = test_data(doc_i, feat_i);
                prod = prod*(w*Ptc(class_i, feat_i) + (1-w)*(1-Ptc(class_i, feat_i)));
                prodn = prodn*(w*Ptnc(class_i, feat_i) + (1-w)*(1-Ptnc(class_i, feat_i)));
            end
            Pcd(class_i, doc_i)  = prod*probcd;
            Pncd(class_i, doc_i) = prodn*probncd;
        end
    end

    % Calculamos el Umbral para cada documento
    % Está dado por el máximo de la muestra - la desviación estándar
    Pthres = zeros(num_class, 1);
    for class_i=1:num_class
        Pthres(class_i, 1) = max(Pcd(class_i, :)) - 2*std2(Pcd(class_i, :));
        %Pthres(class_i, 1) = sum(Pcd(class_i, :))/num_test_docs;
    end
    
    % Aplicamos un umbral a la decision final
    test_predicted = zeros(num_class, num_test_docs);
    for class_i=1:num_class
        for doc_i=1:num_test_docs
            %if(Pcd(class_i, doc_i) >= Pthres)
            if(Pcd(class_i, doc_i) > Pncd(class_i, doc_i))
                test_predicted(class_i, doc_i) = 1;
            end
        end
    end
    
    get_measures(test_predicted, test_target);
    
end


