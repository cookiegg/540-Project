function shuffledData = generateReference()
    frequencies = [];
    for i=1:10
       frequencies(i) = 10*i; 
    end

    ref = [];
    for j=1:length(frequencies)
        ww = 2*pi*frequencies(j);
        ref(:,j) = sin(ww*linspace(0,1,1000))';
    end

    data = [];
    for k=1:10
       data = [data ref]; 
    end
    [n, d] = size(data);

    shuffledData = data(:,randperm(d));
    shuffledData = reshape(shuffledData,[n*d,1]);
    shuffledData = awgn(shuffledData, 10);
end