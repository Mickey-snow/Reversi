import random
from copy import deepcopy

import numpy as np
import OthelloEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############################
# Neural Network Architecture
##############################


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=8, num_channels=128, num_res_blocks=10, num_moves=64):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.num_moves = num_moves
        # Input channels: two channels (current player, opponent)
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=num_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * board_size * board_size, num_moves)

        # Value head
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch, 2, board_size, board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        p = F.log_softmax(p, dim=1)
        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        return p, v


##############################
# Monte Carlo Tree Search (MCTS)
##############################


class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}  # key: move tuple, value: MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def q_value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0


def select(node, c_puct):
    best_score = -float("inf")
    best_move = None
    best_child = None
    for move, child in node.children.items():
        u = child.q_value() + c_puct * child.prior * np.sqrt(node.visit_count) / (
            1 + child.visit_count
        )
        if u > best_score:
            best_score = u
            best_move = move
            best_child = child
    return best_move, best_child


def backpropagate(node, value):
    # Update all nodes on the path with the leaf evaluation.
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        value = -value  # value is from the viewpoint of alternating players
        node = node.parent


def simulate(node, env, model, device, c_puct, root_player):
    curr_node = node
    env_copy = deepcopy(env)
    # Selection phase:
    while curr_node.children and not env_copy.is_game_over():
        move, next_node = select(curr_node, c_puct)
        env_copy.step(move)
        curr_node = next_node
    # Expansion + Evaluation phase:
    if not env_copy.is_game_over():
        state = env_copy.get_state()
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        log_policy, value = model(state_tensor)
        policy = torch.exp(log_policy).detach().cpu().numpy()[0]
        valid_moves = env_copy.valid_moves()
        mask = np.zeros(64, dtype=np.float32)
        for move in valid_moves:
            idx = move[0] * env_copy.board_size + move[1]
            mask[idx] = 1.0
        policy = policy * mask
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = mask / np.sum(mask)
        for move in valid_moves:
            if move not in curr_node.children:
                idx = move[0] * env_copy.board_size + move[1]
                curr_node.children[move] = MCTSNode(parent=curr_node, prior=policy[idx])
        leaf_value = value.item()
    else:
        # Terminal state
        winner = env_copy.get_winner()
        if winner == 0:
            leaf_value = 0
        else:
            leaf_value = 1 if winner == root_player else -1
    backpropagate(curr_node, leaf_value)


def mcts_search(
    root, env, model, device, c_puct=1.0, n_simulations=100, root_player=None
):
    for _ in range(n_simulations):
        simulate(root, env, model, device, c_puct, root_player)


##############################
# Self-Play and Training Data Generation
##############################


def self_play_game(model, device, n_simulations=100, c_puct=1.0):
    env = OthelloEnv()
    training_examples = []
    while not env.is_game_over():
        state = env.get_state()
        current_player = env.current_player
        root = MCTSNode(parent=None, prior=0)
        # Initialize the root children using the network.
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        log_policy, _ = model(state_tensor)
        policy = torch.exp(log_policy).detach().cpu().numpy()[0]
        valid_moves = env.valid_moves()
        mask = np.zeros(64, dtype=np.float32)
        for move in valid_moves:
            idx = move[0] * env.board_size + move[1]
            mask[idx] = 1.0
        policy = policy * mask
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = mask / np.sum(mask)
        for move in valid_moves:
            idx = move[0] * env.board_size + move[1]
            root.children[move] = MCTSNode(parent=root, prior=policy[idx])
        # Run MCTS simulations.
        mcts_search(
            root, env, model, device, c_puct, n_simulations, root_player=current_player
        )
        # Build the move probability (pi) vector based on visit counts.
        move_counts = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_counts.values())
        pi = np.zeros(64, dtype=np.float32)
        for move, count in move_counts.items():
            idx = move[0] * env.board_size + move[1]
            pi[idx] = count / total_visits
        training_examples.append((state, pi, current_player))
        # Choose a move (here probabilistically; during evaluation you might choose argmax)
        moves = list(move_counts.keys())
        counts = list(move_counts.values())
        selected_move = random.choices(moves, weights=counts)[0]
        env.step(selected_move)
    # When the game is over, determine the winner and set the outcome for each training example.
    winner = env.get_winner()
    training_data = []
    for state, pi, player in training_examples:
        if winner == 0:
            outcome = 0
        else:
            outcome = 1 if winner == player else -1
        training_data.append((state, pi, outcome))
    return training_data


##############################
# Training Loop
##############################


def train_model(model, optimizer, training_data, device, batch_size=32, epochs=5):
    model.train()
    loss_fn_value = nn.MSELoss()
    # Shuffle the training data.
    random.shuffle(training_data)
    for epoch in range(epochs):
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]
            states, pis, outcomes = zip(*batch)
            state_tensor = torch.tensor(np.array(states)).to(
                device
            )  # shape: (batch, 2, 8, 8)
            pi_tensor = torch.tensor(np.array(pis)).to(device)  # shape: (batch, 64)
            outcome_tensor = (
                torch.tensor(np.array(outcomes), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            optimizer.zero_grad()
            log_policy_pred, value_pred = model(state_tensor)
            # Compute policy loss using a cross–entropy formulation.
            loss_policy = -torch.mean(torch.sum(pi_tensor * log_policy_pred, dim=1))
            loss_value = loss_fn_value(value_pred, outcome_tensor)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()


def main_loop(
    num_iterations=1000, games_per_iteration=10, n_simulations=100, c_puct=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    all_training_data = []
    for iteration in range(num_iterations):
        iteration_data = []
        for _ in range(games_per_iteration):
            game_data = self_play_game(model, device, n_simulations, c_puct)
            iteration_data.extend(game_data)
        all_training_data.extend(iteration_data)
        train_model(model, optimizer, iteration_data, device, batch_size=32, epochs=5)
        if iteration % 10 == 0:
            print(
                f"Iteration {iteration} completed. Total training examples: {len(all_training_data)}"
            )
    # Optionally, save the final model
    torch.save(model.state_dict(), "alphazero_othello.pth")


if __name__ == "__main__":
    main_loop(num_iterations=100, games_per_iteration=5, n_simulations=100, c_puct=1.0)
