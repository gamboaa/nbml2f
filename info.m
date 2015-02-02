function [ doc_per_cat, doc_in_cat ] = info( train_data, train_target, test_data, test_target )
% Entrega la información sobre los dataset
    [num_class, num_train_docs] = size(train_target);
    [num_train_docs, num_feat] = size(train_data);
    [num_test_docs, num_feat] = size(test_data);
    
    doc_per_cat = zeros(num_class, 1);
    
    for class_i=1:num_class
       doc_per_cat(class_i, 1) = sum(test_target(class_i, :)) + sum(train_target(class_i, :));
    end
    
    % Contiene la cantidad de documentos en 0, 1 o más clasesaaaa
    doc_in_cat = zeros(1, 3);
    for doc_i=1:num_train_docs
        num_cat = sum(train_target(:,doc_i));
        % Documentos que pertenecen a mas de 1 categoría
        if(num_cat > 1)
            doc_in_cat(1, 1) = doc_in_cat(1, 1) + 1;
        end
        
        % Documentos que pertenecen a 1 categoría
        if(num_cat == 1)
            doc_in_cat(1, 2) = doc_in_cat(1, 2) + 1;
        end
        
        % Documentos que pertenecen a 0 categorías
        if(num_cat == 0)
            doc_in_cat(1, 3) = doc_in_cat(1, 3) + 1;
        end
    end
    for doc_i=1:num_test_docs
        num_cat = sum(test_target(:,doc_i));
        % Documentos que pertenecen a mas de 1 categoría
        if(num_cat > 1)
            doc_in_cat(1, 1) = doc_in_cat(1, 1) + 1;
        end
        
        % Documentos que pertenecen a 1 categoría
        if(num_cat == 1)
            doc_in_cat(1, 2) = doc_in_cat(1, 2) + 1;
        end
        
        % Documentos que pertenecen a 0 categorías
        if(num_cat == 0)
            doc_in_cat(1, 3) = doc_in_cat(1, 3) + 1;
        end
    end
end

