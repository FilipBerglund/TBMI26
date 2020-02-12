function [C,E] = StrongClassifier(A,T,P,X,H)

nbrWeakClassifiers = size(A,2);
C = 0;
for t = 1:nbrWeakClassifiers
    C = C + A(t).*WeakClassifier(T(t),P(t),X(H(t),:));
end
C = sign(C);
end