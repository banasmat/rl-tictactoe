import random
import matplotlib.pyplot as plt

NUM_EPISODES = 10000
TEST_EPISODES = 20


class Env:

    X = 'x'
    O = 'o'
    NA = '-'

    size = 3
    win_lines = []

    board = []
    available_actions = []
    player_began_last = False

    def __init__(self, size=3):
        self.size = size
        self.win_lines = self._create_win_lines(size)

    def _create_win_lines(self, size):
        win_lines = []

        diagonal_win_line_1 = []
        diagonal_win_line_2 = []
        vertical_win_lines = {}

        for row in range(0, size):
            horizontal_win_line = []
            for col in range(0, size):
                horizontal_win_line.append((row, col))

                if col not in vertical_win_lines:
                    vertical_win_lines[col] = []
                vertical_win_lines[col].append((row, col))
                if len(vertical_win_lines[col]) == size:
                    win_lines.append(vertical_win_lines[col])

                if row == col:
                    diagonal_win_line_1.append((row, col))
                if row + col == size-1:
                    diagonal_win_line_2.append((row, col))

            win_lines.append(set(horizontal_win_line))

        win_lines.append(set(diagonal_win_line_1))
        win_lines.append(set(diagonal_win_line_2))

        return win_lines

    def reset(self):
        self.available_actions = []
        self.board = []

        for row in range(self.size):
            self.board.append([])
            for col in range(self.size):
                self.board[row].append(self.NA)
                self.available_actions.append((row, col))

        if self.player_began_last:
            env.play_opponents_move()
            self.player_began_last = False
        else:
            self.player_began_last = True

        return self.board

    def observe(self):
        flat_board = [item for sublist in self.board for item in sublist]
        return ''.join(flat_board)

    def step(self, _action, _render=False):

        self._play_on_field(_action, self.X)

        if _render:
            self.render()

        _reward = 0
        occurrences = self.get_win_line_occurrences(self.X)

        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.X]) == 3:
                _reward = 1
                break

        if _reward == 0:
            if len(self.available_actions) == 0:
                _reward = -1
            else:
                self.play_opponents_move()
                occurrences = self.get_win_line_occurrences(self.X)
                for win_line_n, occ_data in occurrences.items():
                    if len(occ_data[self.O]) == 3:
                        _reward = -2
                        break

                if _reward == 0 and len(self.available_actions) == 0:
                    _reward = -1

        _is_done = True if _reward != 0 else False

        if _render:
            self.render()

        return _reward, _is_done

    def _play_on_field(self, field, symbol):
        if self.board[field[0]][field[1]] != self.NA:
            raise Exception("Field already used")

        self.board[field[0]][field[1]] = symbol
        self.available_actions.remove(field)

    def play_opponents_move(self):
        occurrences = self.get_win_line_occurrences(self.O)
        # win if it's possible
        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.O]) == 2 and len(occ_data[self.NA]) == 1:
                play_field = occ_data[self.NA][0]
                return self._play_on_field(play_field, self.O)
        # block player
        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.X]) == 2 and len(occ_data[self.NA]) == 1:
                play_field = occ_data[self.NA][0]
                return self._play_on_field(play_field, self.O)
        # continue building row
        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.O]) > 0 and len(occ_data[self.NA]) > 0 and len(occ_data[self.X]) == 0:
                play_field = occ_data[self.NA][random.randint(0, len(occ_data[self.NA])-1)]
                return self._play_on_field(play_field, self.O)
        # start building new row
        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.NA]) == 3:
                play_field = occ_data[self.NA][random.randint(0,2)]
                return self._play_on_field(play_field, self.O)
        # if none of above is possible, play random
        return self._play_on_field(self.sample_action(), self.O)

    def get_win_line_occurrences(self, symbol):

        occurrences = {}

        for i, win_line in enumerate(self.win_lines):
            occurrences[i] = {}
            occurrences[i][self.X] = []
            occurrences[i][self.O] = []
            occurrences[i][self.NA] = []

            for win_field in win_line:
                for symbol in [self.X, self.O, self.NA]:
                    if self.board[win_field[0]][win_field[1]] == symbol:
                        occurrences[i][symbol].append(win_field)

        return occurrences

    def sample_action(self):
        return random.choice(self.available_actions)

    def render(self):
        print(self.board[0])
        print(self.board[1])
        print(self.board[2])
        print()


class Agent:

    GAMMA = 0.9
    START_EPSILON = 1.0
    END_EPSILON = 0.2
    EPSILON_DECAY = 0.001

    q_history = []
    epsilon = START_EPSILON
    q_map = {}

    def play_episode(self, _env: Env, _render=False):

        if random.randint(0, 1000)/1000 > self.epsilon:
            _action = self.choose_best_action(_env)
        else:
            _action = _env.sample_action()

        self.q_history.append((_env.observe(), _action))

        _reward, _is_done = _env.step(_action, render)

        if _is_done:
            self.update_q_vals(_env, _reward)
            self.q_history = []

        self.epsilon -= self.EPSILON_DECAY

        return _reward, _is_done

    def choose_best_action(self, _env: Env):
        _state = _env.observe()

        q_vals = {}
        for act in env.available_actions:
            if (_state, act) not in self.q_map:
                self.q_map[_state, act] = 0.0
            q_vals[act] = self.q_map[_state, act]

        max_acts = []
        for _act, val in q_vals.items():
            if val == max(q_vals.values()):
                max_acts.append(_act)

        return random.choice(max_acts)

    def update_q_vals(self, _env, _reward):

        prev_val = _reward

        for _state, _action in reversed(self.q_history):

            if (_state, _action) not in self.q_map:
                self.q_map[_state, _action] = 0.0

            self.q_map[_state, _action] = 0.8 * self.q_map[_state, _action] + 0.2 * self.GAMMA * prev_val
            prev_val = self.q_map[_state, _action]


if __name__ == "__main__":
    env = Env()
    agent = Agent()
    count_episodes = 0
    total_reward = 0
    mean_rewards = []

    for i in range(0, NUM_EPISODES):

        env.reset()

        while True:

            render = False
            if i > NUM_EPISODES-10:
                render = True

            reward, is_done = agent.play_episode(env, render)

            if is_done:
                total_reward += reward
                count_episodes += 1

                # print('episode: %i, total reward: %i' % (i, total_reward))
                # env.render()
                break

        if i % TEST_EPISODES == 0:
            mean_reward = total_reward/count_episodes
            # print('episode: %i, mean reward: %f.3' % (i, mean_reward))
            mean_rewards.append(mean_reward)
            # env.render()
            total_reward = 0
            count_episodes = 0

    # for st, act in agent.q_map:
    #     if st == 'o-ox-----':
    #     # if st == 'o--------':
    #         print(st, act, agent.q_map[st, act])

    plt.plot(mean_rewards)
    plt.show()
