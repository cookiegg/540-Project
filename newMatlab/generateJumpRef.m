function jumpRef = generateJumpRef()
    % expect number of jumps is timeSteps/avgHold = 600000/60 = about 10000
    jumpRef = [];
    timeSteps = 1000000;
    minHold = 10;
    maxHold = 100;
    minJump = 0.2;
    maxJump = 3;
    sign = [-1 1];
    while(length(jumpRef) < timeSteps)
        r = randi([minHold, maxHold]);
        value = (maxJump-minJump)*rand + minJump;
        value = value*sign(randi(length(sign))); % +/-
        % value = randi([minJump, maxJump]);
        jumpRef = [jumpRef; value*ones(r,1)];
    end
    figure;
    plot(jumpRef);
    axis([0,1000,-3.5,3.5]);
end