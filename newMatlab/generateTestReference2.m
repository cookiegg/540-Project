function ref = generateTestReference2()
    frequency = 10;
    ww = 2*pi*frequency;
    sinw= sin(ww*linspace(0,1,200));
    z = [0];
    x = linspace(0,1,200);
    ref = ones(1,200);
    for i=1:length(z)
        ref = ref.*(x-z(i));
    end
    ref = -3*ref/(max(abs(ref)));
    wtf = ref;
    for i=1:4
       wtf = [wtf ref]; 
    end
    %ref = ref + 0.3*sinw;
    ref = wtf;
    plot(ref);
    ref = ref';
end