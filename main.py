import random

NUM_EPISODES = 10

class Env:

    X = 'x'
    O = 'o'
    NA = '-'

    ROWS = 3
    COLS = 3
    WIN_LINES = [
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
        ((2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 1), (1, 1), (2, 1)),
        ((0, 2), (1, 2), (2, 2)),
        ((0, 0), (1, 1), (2, 2)),
        ((0, 2), (1, 1), (2, 0)),
    ]

    board = []
    state_history = []
    available_actions = []

    def reset(self):
        self.available_actions = []
        self.state_history = []
        self.board = []

        for row in range(self.ROWS):
            self.board.append([])
            for col in range(self.COLS):
                self.board[row].append(self.NA)
                self.available_actions.append((row, col))

        if random.randint(0, 1) == 1:
            env.play_opponents_move()

        return self.board

    def step(self, _action):

        self._play_on_field(_action, self.X)
        self.state_history.append(self.board)

        _reward = 0
        occurrences = self.get_win_line_occurrences(self.X)

        for win_line_n, occ_data in occurrences.items():
            if len(occ_data[self.X]) == 3:
                _reward = 1
                break

        if len(self.available_actions) == 0:
            _reward = -1
        else:
            self.play_opponents_move()

            for win_line_n, occ_data in occurrences.items():
                if len(occ_data[self.O]) == 3:
                    _reward = -2
                    break

            if len(self.available_actions) == 0:
                _reward = -1

        _is_done = True if _reward != 0 else False

        return _reward, self.board, _is_done

    def _play_on_field(self, field, symbol):
        if self.board[field[0]][field[1]] != self.NA:
            raise Exception("Field already used")

        self.board[field[0]][field[1]] = symbol
        self.available_actions.remove(field)

    def play_opponents_move(self):
        occurrences = self.get_win_line_occurrences(self.O)
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

        for i, win_line in enumerate(self.WIN_LINES):
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


env = Env()
count_episodes = 0
total_reward = 0

for i in range(0, NUM_EPISODES):

    env.reset()
    count_episodes += 1

    # while True:
    for j in range(0, 9):
        action = env.sample_action()

        reward, next_state, is_done = env.step(action)

        if is_done:
            total_reward += reward

            print('episode: %i, total reward: %i' % (i, total_reward))
            env.render()
            break

    # if i % 10 == 0:
    #     print('episode: %i, total reward: %i' % (i, total_reward))
    #     env.render()

