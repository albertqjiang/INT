import json
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from legacy.helper_functions.QNetwork import GeneralQNetwork
from legacy.helper_functions.utils import *
from logic.logic import Proof
from legacy.logic.logicRL import LogicBasedProver
from legacy.logic_math import real_number_axioms
from legacy.helper_functions.RecursiveAttentiveEncoder import *


def test_agent(this_encoder, lower_q_net, higher_q_net, list_of_proof_and_provers, degree, max_steps, device,
               lower_reward_scheme, higher_reward_scheme):
    higher_rewards = list()
    lower_rewards = list()
    optimality = list()
    list_of_proofs = list()
    for proof, prover in list_of_proof_and_provers:
        this_proof, this_prover = deepcopy(proof), deepcopy(prover)
        total_higher_reward = 0
        total_lower_reward = 0
        lower_step = 0
        for outer_step in range(1, 1 + max_steps):
            if (len(this_proof.ground_truth) > this_encoder.ground_truth_maxsize - 2) or \
                    (len(this_proof.lemmas) > this_encoder.theorem_maxsize - 1) or \
                    (len(this_proof.entities) > this_encoder.entity_maxsize - 5):
                outer_step = max_steps
                break
            list_of_entities = list(this_proof.entities)
            lower_action_space = list_of_entities
            higher_action_space = this_proof.lemmas
            lower_action_tensor_space = torch.cat([this_encoder.encode_entity(entity) for entity in list_of_entities],
                                                  dim=0)
            higher_action_tensor_space = torch.cat([this_encoder.encode_theorem(lemma) for lemma in this_proof.lemmas],
                                                   dim=0)

            higher_state = this_encoder.higher_attentive_state(this_prover.raw_observe())
            lemma_chosen = higher_q_net.sample_action(state=higher_state, action_space=higher_action_space,
                                                      action_tensor_space=higher_action_tensor_space)[0]
            sequence_of_operands = list()
            for operand_index in range(1, 1 + lemma_chosen.input_no):
                # Lower MDP and update
                lower_state = this_encoder.lower_attentive_state(observation=this_prover.raw_observe(),
                                                                 lemma_chosen=lemma_chosen,
                                                                 sequence_of_entities=sequence_of_operands)
                entity_chosen = lower_q_net.sample_action(state=lower_state, action_space=lower_action_space,
                                                          action_tensor_space=lower_action_tensor_space)[0]
                sequence_of_operands.append(entity_chosen)
                lower_reward, reward_string = this_proof.apply_theorem_get_reward(theorem=lemma_chosen,
                                                                                  operands=sequence_of_operands,
                                                                                  reward_scheme=lower_reward_scheme)
                total_lower_reward += lower_reward
                lower_step += 1

            # Higher MDP and update
            higher_reward = higher_reward_scheme[reward_string]
            total_higher_reward += higher_reward

            if this_proof.proved:
                break
        higher_rewards.append(total_higher_reward / outer_step)
        lower_rewards.append(total_lower_reward / (lower_step - 0.001))
        optimality.append(outer_step / degree)
        if this_proof.proved:
            list_of_proofs.append(this_proof.print_proof_status())
        else:
            list_of_proofs.append("Didn't manage to prove:\n")
            list_of_proofs.append(this_proof.print_proof_status())
    return higher_rewards, lower_rewards, optimality, list_of_proofs


def update(q_net, optimizer, action_space, action_tensor_space, buffer, configuration):
    batch_size = configuration["batch_size"]
    discount_factor = configuration["discount_factor"]
    list_of_batches = prepare_batches(buffer, batch_size)
    for batch in list_of_batches:
        dict_batch = convert_batch_dict_to_dict_batch(batch)
        state = dict_batch["state"]
        action = dict_batch["action"]
        next_state = dict_batch["next_state"]
        reward = dict_batch["reward"]
        done = dict_batch["done"]
        loss = q_net.q_loss(state=state, action=action, reward=reward, next_state=next_state, done=done,
                            action_space=action_space, action_tensor_space=action_tensor_space,
                            discount_factor=discount_factor)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


def objectives_to_coupled_proofs_and_provers(objectives, encoder, simple_proof):
    proof_prover_couples = list()
    for objective in objectives:
        couple_proof = Proof(entities=simple_proof.initial_entities, assumptions=simple_proof.assumptions,
                             axioms=[axiom for axiom in real_number_axioms.values()], objectives=[objective])
        couple_prover = LogicBasedProver(
            proof=couple_proof, ent_maxsize=encoder.entity_maxsize, gt_maxsize=encoder.ground_truth_maxsize,
            lemma_maxsize=encoder.theorem_maxsize, objective_maxsize=encoder.objective_maxsize,
            lemma_operand_size=encoder.theorem_operand_maxsize, lemma_embedding_size=encoder.theorem_embedding_size,
        )
        proof_prover_couples.append((couple_proof, couple_prover))
    return proof_prover_couples


def train_one_degree(max_frames, max_steps, lower_q_net, lower_q_optimizer, higher_q_net, higher_q_optimizer,
                     encoder, lower_buffer, higher_buffer, config, training_info,
                     degree=1, random_seed=0):
    training_trigger = False
    frame = 0

    # Load stuff
    lower_reward_scheme = config["lower_reward_dict"]
    higher_reward_scheme = config["higher_reward_dict"]
    batch_size = config["batch_size"]
    objective_format = config["objective_format"]
    number_of_objectives = config["number_of_objectives"]
    objective_path = objective_format.format(degree, number_of_objectives)
    objectives = pickle.load(open(objective_path, "rb"))
    simple_proof_path = config["simple_proof_path"]
    saving_dir = config["saving_dir"]
    saving_affix = config["saving_affix"]
    device = encoder.device
    simple_proof = pickle.load(open(simple_proof_path, "rb"))
    proof_prover_couples = objectives_to_coupled_proofs_and_provers(objectives, encoder, simple_proof, )

    # Initialize training info
    if degree not in training_info:
        training_info[degree] = {
            "lower_rewards": list(),
            "higher_rewards": list(),
            "optimality": list(),
            "proving_process": list(),
        }

    while frame < max_frames:
        print(frame)
        if frame > 0.1 * int(max_frames) and "10%" not in training_info[degree]:
            training_info[degree]["10%"] = test_agent(
                degree=degree, device=device, this_encoder=encoder, higher_q_net=higher_q_net,
                lower_q_net=lower_q_net, higher_reward_scheme=higher_reward_scheme,
                lower_reward_scheme=lower_reward_scheme, list_of_proof_and_provers=proof_prover_couples,
                max_steps=max_steps
            )
        if frame > 0.5 * int(max_frames) and "50%" not in training_info[degree]:
            training_info[degree]["50%"] = test_agent(
                degree=degree, device=device, this_encoder=encoder, higher_q_net=higher_q_net,
                lower_q_net=lower_q_net, higher_reward_scheme=higher_reward_scheme,
                lower_reward_scheme=lower_reward_scheme, list_of_proof_and_provers=proof_prover_couples,
                max_steps=max_steps
            )
        if frame > 0.9 * int(max_frames) and "90%" not in training_info[degree]:
            training_info[degree]["90%"] = test_agent(
                degree=degree, device=device, this_encoder=encoder, higher_q_net=higher_q_net,
                lower_q_net=lower_q_net, higher_reward_scheme=higher_reward_scheme,
                lower_reward_scheme=lower_reward_scheme, list_of_proof_and_provers=proof_prover_couples,
                max_steps=max_steps
            )

        proof, prover = deepcopy(random.choice(proof_prover_couples))
        aggregate_lower_reward = 0
        aggregate_lower_step = 0
        aggregate_higher_reward = 0
        lower_success_trials = 0
        lower_total_trials = 0

        for step in range(1, 1 + max_steps):
            # TODO: This is a hack, should get rid of it in the future
            if (len(proof.ground_truth) > encoder.ground_truth_maxsize - 2) or \
                    (len(proof.lemmas) > encoder.theorem_maxsize - 1) or \
                    (len(proof.entities) > encoder.entity_maxsize - 5):
                step = max_steps
                break
            list_of_entities = list(proof.entities)
            lower_action_space = list_of_entities
            higher_action_space = proof.lemmas
            lower_action_tensor_space = torch.cat([encoder.encode_entity(entity) for entity in list_of_entities], dim=0)
            higher_action_tensor_space = torch.cat([encoder.encode_theorem(lemma) for lemma in proof.lemmas], dim=0)

            higher_state = encoder.higher_attentive_state(prover.raw_observe())
            lemma_chosen = higher_q_net.sample_action(state=higher_state, action_space=higher_action_space,
                                                      action_tensor_space=higher_action_tensor_space)[0]
            lemma_tensor = encoder.encode_theorem(lemma_chosen)
            sequence_of_operands = list()
            for operand_index in range(1, 1 + lemma_chosen.input_no):
                # Lower MDP and update
                lower_state = encoder.lower_attentive_state(observation=prover.raw_observe(), lemma_chosen=lemma_chosen,
                                                            sequence_of_entities=sequence_of_operands)
                entity_chosen = lower_q_net.sample_action(state=lower_state, action_space=lower_action_space,
                                                          action_tensor_space=lower_action_tensor_space)[0]
                entity_chosen_tensor = encoder.encode_entity(entity_chosen)
                sequence_of_operands.append(entity_chosen)
                lower_next_state = encoder.lower_attentive_state(
                    observation=prover.raw_observe(), lemma_chosen=lemma_chosen,
                    sequence_of_entities=sequence_of_operands
                )
                lower_reward, reward_string = proof.apply_theorem_get_reward(theorem=lemma_chosen,
                                                                             operands=sequence_of_operands,
                                                                             reward_scheme=lower_reward_scheme)
                aggregate_lower_reward += lower_reward
                aggregate_lower_step += 1
                lower_reward = torch.FloatTensor([[lower_reward]]).to(device)
                lower_done = 1 if (operand_index == lemma_chosen.input_no) else 0
                lower_done = torch.FloatTensor([[lower_done]]).to(device)
                lower_buffer.push(state=lower_state, next_state=lower_next_state, reward=lower_reward,
                                  action=entity_chosen_tensor, done=lower_done)
                if len(lower_buffer) >= batch_size and training_trigger:
                    update(q_net=lower_q_net, optimizer=lower_q_optimizer, action_space=lower_action_space,
                           action_tensor_space=lower_action_tensor_space,
                           buffer=lower_buffer, configuration=config)
            if training_trigger:
                frame += 1
            if lower_reward > 0:
                lower_success_trials += 1
            lower_total_trials += 1

            # Higher MDP and update
            higher_next_state = encoder.higher_attentive_state(prover.raw_observe())
            higher_done = 1 if proof.proved else 0
            higher_done = torch.FloatTensor([[higher_done]]).to(device)
            higher_reward = higher_reward_scheme[reward_string]
            aggregate_higher_reward += higher_reward
            higher_reward = torch.FloatTensor([[higher_reward]]).to(device)
            higher_buffer.push(state=higher_state, next_state=higher_next_state, reward=higher_reward,
                               action=lemma_tensor, done=higher_done)
            if len(higher_buffer) >= batch_size and training_trigger:
                update(q_net=higher_q_net, optimizer=higher_q_optimizer, action_space=higher_action_space,
                       action_tensor_space=higher_action_tensor_space, buffer=higher_buffer, configuration=config)

            if proof.proved:
                training_trigger = True
                training_info[degree]["proving_process"].append(proof.print_proof_status())
                break

        if frame >= max_frames - 1 and "100%" not in training_info[degree]:
            training_info[degree]["100%"] = test_agent(
                degree=degree, device=device, this_encoder=encoder, higher_q_net=higher_q_net,
                lower_q_net=lower_q_net, higher_reward_scheme=higher_reward_scheme,
                lower_reward_scheme=lower_reward_scheme, list_of_proof_and_provers=proof_prover_couples,
                max_steps=max_steps
            )

        if aggregate_lower_reward == 0 and aggregate_lower_step == 0:
            training_info[degree]["lower_rewards"].append(0)
        elif aggregate_lower_reward != 0 and aggregate_lower_step == 0:
            print(aggregate_lower_reward, aggregate_lower_step)
        else:
            training_info[degree]["lower_rewards"].append(aggregate_lower_reward / aggregate_lower_step)
        training_info[degree]["higher_rewards"].append(aggregate_higher_reward / step)
        training_info[degree]["optimality"].append(step / degree)
        json.dump(training_info,
                  open("{}training_info{}_seed{}.json".format(saving_dir, saving_affix, random_seed), "w"))
        model = {
            "higher_q_net": higher_q_net, "higher_q_optimizer": higher_q_optimizer,
            "lower_q_net": lower_q_net, "lower_q_optimizer": lower_q_optimizer,
            "training_info": training_info, "encoder": encoder
        }
        torch.save(model, "{}checkpoint{}_seed{}.pt".format(saving_dir, saving_affix, random_seed))


def train(random_seed):
    set_random_seed(random_seed)
    # Loading stuff
    config = load_config()["hier_q_curriculum"]
    higher_lr = config["higher_lr"]
    lower_lr = config["lower_lr"]
    capacity = config["replay_buffer_capacity"]
    saving_dir = config["saving_dir"]
    saving_affix = config["saving_affix"]
    max_frames = config["max_frames"]
    max_steps = config["max_steps"]
    max_degree = config["max_degree"]

    if config["pretrain"] == "True":
        encoder = torch.load(config["encoder_path"])
    elif config["pretrain"] == "False":
        encoder_config = load_config()["recur"]
        proof = pickle.load(open("../data/simple_proof.p", "rb"))
        prover = LogicBasedProver(
            proof=proof,
            ent_maxsize=encoder_config["ent_maxsize"],
            gt_maxsize=encoder_config["gt_maxsize"],
            lemma_maxsize=encoder_config["lemma_maxsize"],
            lemma_embedding_size=encoder_config["lemma_embedding_size"],
            lemma_operand_size=encoder_config["lemma_operand_size"],
            objective_maxsize=encoder_config["objective_maxsize"]
        )
        encoder = RecursiveAttentiveEncoder(prover)

    # for encoder_para in encoder.parameters():
    #     encoder_para.requires_grad = False
    lower_buffer = ReplayBuffer("L1 Buffer", capacity)
    higher_buffer = ReplayBuffer("L2 Buffer", capacity)

    # Figuring out dimensions
    state_dim = encoder.observation_tensor_size
    higher_action_dim = encoder.theorem_embedding_size
    lower_action_dim = encoder.entity_embedding_size

    # Initializing nets and optimizers
    print(state_dim + higher_action_dim + encoder.theorem_operand_maxsize * lower_action_dim, lower_action_dim)
    print(state_dim, higher_action_dim)
    lower_q_net = GeneralQNetwork(
        state_dim=encoder.lower_attentive_size,
        action_dim=lower_action_dim
    )
    lower_q_optimizer = torch.optim.Adam(list(lower_q_net.parameters()) + list(encoder.parameters()),
                                         lr=lower_lr, amsgrad=True)
    higher_q_net = GeneralQNetwork(state_dim=encoder.higher_attentive_size, action_dim=higher_action_dim)
    higher_q_optimizer = torch.optim.Adam(higher_q_net.parameters(), lr=higher_lr, amsgrad=True)

    if os.path.isfile("{}checkpoint{}_seed{}.pt".format(saving_dir, saving_affix, random_seed)):
        model = torch.load("{}checkpoint{}_seed{}.pt".format(saving_dir, saving_affix, random_seed))
        higher_q_net = model["higher_q_net"]
        higher_q_optimizer = model["higher_q_optimizer"]
        lower_q_net = model["lower_q_net"]
        lower_q_optimizer = model["lower_q_optimizer"]
        training_info = model["training_info"]
        encoder = model["encoder"]
    else:
        training_info = dict()

    for degree in range(1, 1 + max_degree):
        train_one_degree(
            max_frames=max_frames, max_steps=max_steps,
            lower_q_net=lower_q_net, lower_q_optimizer=lower_q_optimizer,
            higher_q_net=higher_q_net, higher_q_optimizer=higher_q_optimizer,
            encoder=encoder,
            lower_buffer=lower_buffer, higher_buffer=higher_buffer,
            config=config, training_info=training_info, random_seed=random_seed,
            degree=degree
        )


if __name__ == "__main__":
    total_random_seeds = 1
    for seed in range(total_random_seeds):
        train(seed)
