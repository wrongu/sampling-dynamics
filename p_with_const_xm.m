syms p;
targets = [.7,.8,.9];
deps = [p, 1-p];

ps = zeros(6,3);

for ti=1:3
    target = targets(ti);
    for m = 2:7
        poly = sym([1, 0]);
        for k=1:m
            % P(xk=0) = P(xk=0|xj=0)P(xj=0) + P(xk=0|xj=1)P(xj=1)
            %         = deps(1)  *  poly(1  + deps(2)  *  poly(2)
            new_poly = ...
                [poly(1) *  deps(1) + poly(2) * deps(2), ...
                 poly(2) * deps(1) + poly(1) * deps(2)];
             poly = new_poly;
        end
        soln = roots(sym2poly(poly(1)-target));
        soln = real(soln(1));
        ps(m-1,ti) = soln;
        fprintf('m = %d\n', m);
        fprintf('\tp = %f\n', soln);
    end
end

figure()
plot(2:7,ps);
legend({'P(X_m|X_1)=.7','P(X_m|X_1)=.8','P(X_m|X_1)=.9'});
xlabel('model depth');
ylabel('p (consistency strenth)');