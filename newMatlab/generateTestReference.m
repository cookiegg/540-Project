function ref = generateTestReference()
    frequency = 10;
    ww = 2*pi*frequency;
    sinw= sin(ww*linspace(0,1,1000));
    z = [0 0.3 0.4 0.5 0.8 1];
    x = linspace(0,1,1000);
    ref = ones(1,1000);
    for i=1:length(z)
        ref = ref.*(x-z(i));
    end
    ref = ref/(max(abs(ref)));
    ref = ref + 0.3*sinw;
    plot(ref);
    ref = ref';
end