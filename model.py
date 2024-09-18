import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_inputs[0], out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(hidden_dim * num_inputs[1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the conv layer output
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.conv1_q1 = nn.Conv1d(in_channels=num_inputs[0], out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2_q1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear1_q1 = nn.Linear(hidden_dim * num_inputs[1] + num_actions[0] * num_actions[1], hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.conv1_q2 = nn.Conv1d(in_channels=num_inputs[0], out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2_q2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear1_q2 = nn.Linear(hidden_dim * num_inputs[1] + num_actions[0] * num_actions[1], hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu1 = F.relu(self.conv1_q1(state))
        xu1 = F.relu(self.conv2_q1(xu1))
        #print(f"BEFORE VIEW: {xu1.shape=}")
        xu1 = xu1.view(xu1.size(0), -1)  # Flatten the conv layer output
        #print(f"AFTER VIEW: {xu1.shape=}")
        xu1 = torch.cat([xu1, action.view(action.size(0), -1)], 1)
        #print(f"AFTER CAT: {xu1.shape=}")
        xu1 = F.relu(self.linear1_q1(xu1))
        #print(f"AFTER RELU: {xu1.shape=}") 
        x1 = self.linear2_q1(xu1)
        #print(f"X1 linear: {x1.shape=}")

        xu2 = F.relu(self.conv1_q2(state))
        xu2 = F.relu(self.conv2_q2(xu2))
        xu2 = xu2.view(xu2.size(0), -1)  # Flatten the conv layer output
        xu2 = torch.cat([xu2, action.view(action.size(0), -1)], 1)
        xu2 = F.relu(self.linear1_q2(xu2))
        x2 = self.linear2_q2(xu2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv1d(in_channels=num_inputs[0], out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(hidden_dim * num_inputs[1], hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions[0] * num_actions[1])
        self.log_std_linear = nn.Linear(hidden_dim, num_actions[0] * num_actions[1])

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).view(1, -1)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).view(1, -1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the conv layer output
        x = F.relu(self.linear1(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        #print(f"SAMPLE-STATE: {state.shape=}-{state}")
        mean, log_std = self.forward(state)
        std = log_std.exp()
        #print(f"SAMPLE-mean: {mean.shape=}-{mean}\n{std.shape=}-{std}")
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        #print(f"SAMPLE-xt: {x_t.shape=}-{x_t}\n{y_t.shape=}-{y_t}")
        action = y_t * self.action_scale + self.action_bias
        #print(f"SAMPLE-ACTION: {action.shape=}-{action}")
        log_prob = normal.log_prob(x_t)
        #print(f"SAMPLE_LOGPORB 1 : {log_prob.shape=}-{log_prob}")
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        #print(f"SAMPLE_LOGPROB 2: {log_prob.shape=}-{log_prob}")
        log_prob = log_prob.mean(1, keepdim=True)
        #print(f"SAMPLE LOGPROB SUM: {log_prob.shape=}-{log_prob}")
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.view(state.size(0), self.num_actions[0], self.num_actions[1]), log_prob, mean.view(state.size(0), self.num_actions[0], self.num_actions[1])

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_inputs[0], out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(hidden_dim * num_inputs[1], hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions[0] * num_actions[1])
        self.noise = torch.Tensor(num_actions[0], num_actions[1])

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).view(1, -1)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).view(1, -1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the conv layer output
        x = F.relu(self.linear1(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean.view(state.size(0), -1, 512)

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)