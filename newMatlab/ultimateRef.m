function ref = ultimateRef()
    x = generateJumpRef();
    x(1000001:end) = [];
    y = generateReference();
    ref = x + y;
    figure;
    plot(ref);
    axis([0,1000,-5.5,5.5]);
end