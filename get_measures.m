function [Pres, Rec] = get_measures( test_predicted, test_target )
% Recibe la matriz de salida del clasificador y
% la compara con la matriz target, entregando los resultados
% Params
% predicted : Matriz con los resultados de los dicumentos evaluados
% target    : Matriz objetivo
    
    % Obtenemos la cantidad de Categorias y documentos
    [num_cat, num_test_docs] = size(test_target);

    fprintf('MICRO Measures\n');
    fprintf('Precision\tRecall\tMicro-F\n');
    
    % MEDIDAS DE DESEMPEÑO
    incorrect_labels = 0;
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    for cat_i=1:num_cat
        for doc_i=1:num_test_docs
            if(test_predicted(cat_i, doc_i) ~= test_target(cat_i, doc_i))
                incorrect_labels = incorrect_labels + 1;
            end
            if (test_predicted(cat_i, doc_i) == 1 && test_target(cat_i, doc_i) == 1)
                TP = TP + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 1 && test_target(cat_i, doc_i) == 0)
                FP = FP + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 0 && test_target(cat_i, doc_i) == 0)
                TN = TN + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 0 && test_target(cat_i, doc_i) == 1)
                FN = FN + 1; 
            end
        end
    end
    hloss = incorrect_labels/(num_cat*num_test_docs);
    precision = TP/(TP + FP);
    recall = TP/(TP + FN);
    F = (2*precision*recall)/(recall + precision);
    fprintf('%.4f\t\t%.4f\t\t%.4f\n', precision, recall, F);
    
    fprintf('%.4f\n', hloss);
    
    % MEDIDAS DE DESEMPEÑO 2
    Fi = 0;
    TPi = 0;
    TNi = 0;
    FPi = 0;
    FNi = 0;
    
    Pres = zeros(num_cat, 1);
    Rec = zeros(num_cat, 1);
    
    for cat_i=1:num_cat
        for doc_i=1:num_test_docs
            if (test_predicted(cat_i, doc_i) == 1 && test_target(cat_i, doc_i) == 1)
                TPi = TPi + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 1 && test_target(cat_i, doc_i) == 0)
                FPi = FPi + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 0 && test_target(cat_i, doc_i) == 0)
                TNi = TNi + 1; 
            end
            if (test_predicted(cat_i, doc_i) == 0 && test_target(cat_i, doc_i) == 1)
                FNi = FNi + 1; 
            end
        end
        Pres(cat_i, 1) = TPi/(TPi + FPi);
        Rec(cat_i, 1) = TPi/(TPi + FNi);
        if(isnan(Pres(cat_i, 1)))
            Pres(cat_i, 1) = 1;
        end
        if(isnan(Rec(cat_i, 1)))
            Rec(cat_i, 1) = 1;
        end
        TPi = 0;
        TNi = 0;
        FPi = 0;
        FNi = 0;
    end
    precisionM = sum(Pres(:, 1))/num_cat;
    recallM = sum(Rec(:, 1))/num_cat;
    FM = (2*precisionM*recallM)/(recallM + precisionM);
    fprintf('MACRO Measures\n');
    fprintf('Precision\tRecall\tMacro-F\n');
    fprintf('%.4f\t\t%.4f\t\t%.4f\n', precisionM, recallM, FM);
end

