function shuffledData = generateReference()
    frequencies = [];
    for i=1:100
       num = i/10;
       frequencies(i) = 10*num; 
    end

    ref = [];
    for j=1:length(frequencies)
        ww = 2*pi*frequencies(j);
        ref(:,j) = 3*sin(ww*linspace(0,1,1000))';
    end

    data = [];
    for k=1:10
       data = [data ref]; 
    end
    [n, d] = size(data);

    shuffledData = data(:,randperm(d));
    shuffledData = reshape(shuffledData,[n*d,1]);
    shuffledData = awgn(shuffledData, 10);
    
    figure;
    plot(shuffledData);
    axis([0,1000,-3.5,3.5]);
end