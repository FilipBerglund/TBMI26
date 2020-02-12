function E = StrongClassifierError(C, Y)
E = sum(Y ~= C)/size(Y,2);
end