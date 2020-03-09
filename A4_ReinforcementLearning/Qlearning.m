%% Initialization
%  Initialize the world, Q-table, and hyperparameters
maxEpisodes = 2500;
learningRate = 0.3;
discountFactor = 0.9;
World = 4;
world = gwinit(World);
Q = zeros(world.ysize,world.xsize,4);
for i = 1:size(Q,1)
    Q(i,1,4) = -Inf;
end
for i = 1:size(Q,1)
    Q(i,end,3) = -Inf;
end
for i = 1:size(Q,2)
    Q(1,i,2) = -Inf;
end
for i = 1:size(Q,2)
    Q(end,i,1) = -Inf;
end

%% Training loop
%  Train the agent using the Q-learning algorithm.
for episode = 1:maxEpisodes
    gwinit(World);
    state = gwstate();
    while state.isterminal == 0
        old_pos = state.pos;
        action = chooseaction(Q,state.pos(1),state.pos(2),[1 2 3 4],[1 1 1 1], 0.8);
        state = gwaction(action);
        Q(old_pos(1),old_pos(2),action) = ...
            (1-learningRate)*Q(old_pos(1),old_pos(2),action) ...
            + learningRate*(state.feedback ...
            + discountFactor*(max(Q(state.pos(1),state.pos(2),:))));
    end
    if mod(episode,500) == 0
        fprintf("%d", episode)
    end
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
%world = gwinit(World);
state = gwstate();
while ~state.isterminal
        [~, optAction] = chooseaction(Q,state.pos(1),state.pos(2),[1 2 3 4],[1 1 1 1],0.8);
        state = gwaction(optAction);
        gwdraw()
        pause(0.1)
end
figure(1)
gwdraw()
P = getpolicy(Q);
gwdrawpolicy(P);
figure(2)
imagesc(getvalue(Q), [-1,0]);
colorbar;

%%
world = gwinit(4);
P = getpolicy(Q);
gwdraw([],P)

