

function network = initial_network(init_para)

disp('initial network.')

% random group clusters in CPL
network.weight_recurrent_CPL = randperm(init_para.numNeurons_CPL);

% set weights between input and CLP
if init_para.flag_herg_inputCPL
    row = init_para.numNeurons_CPL;
    col = init_para.numNeurons_input;
    network.weight_input_CPL = zeros(row, col);
    num_prob = numel(init_para.hprob_input_CPL);
    for i = 1:init_para.numNeurons_cluster:init_para.numNeurons_CPL
        ind_cluster = network.weight_recurrent_CPL(i:i+init_para.numNeurons_cluster-1);
        prob = init_para.hprob_input_CPL(randi(num_prob));
        
        for k = 1:init_para.numNeurons_cluster
            network.weight_input_CPL(ind_cluster(k), :) = sprand(1, col,prob);
        end
    end
else
    network.weight_input_CPL = full(sprand(init_para.numNeurons_CPL, init_para.numNeurons_input,...
                       init_para.prob_input_CPL));
end
network.weight_input_CPL(find(network.weight_input_CPL>0.5)) = 1;


% set the output weights
network.weight_CPL_decision = randi([init_para.low_decision_weight, init_para.high_decision_weight],...
                                init_para.numNeurons_decision, init_para.numNeurons_CPL);
network.weightFilter_CPL_decision = tanh(network.weight_CPL_decision/init_para.bound_decision_weight);

