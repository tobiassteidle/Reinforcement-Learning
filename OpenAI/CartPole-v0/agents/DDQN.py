from memory import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *

class DDQNAgent(object):
    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size=10000,
                 batch_size=32,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.1,
                 epsilon_steps_to_min=1000,
                 tau=0.1,
                 mode='QNetwork',
                 use_PER=True,
                 pre_trained=None):

        self.state_size = state_size
        self.action_size = action_size


        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_step = (self.epsilon - self.epsilon_min) / epsilon_steps_to_min
        self.tau = tau

        self.model = self.build_model(mode, pre_trained)
        self.target_model = self.build_model(mode, pre_trained)
        self.hard_update_target_network()

        self.use_PER = use_PER

        if self.use_PER:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = Memory(max_size=buffer_size)

    def build_model(self, mode, pre_trained):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))

        if mode == "QNetwork":
            model.add(Dense(self.action_size, activation='linear'))

        if mode == "DuelingDQN":
            model.add(Dense(self.action_size + 1, activation='linear'))
            model.add(Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True),
                             output_shape=(self.action_size,)))

        if pre_trained:
            model = load_model(pre_trained)

        model.compile(optimizer=Adam(lr=0.001), loss='mse')
        return model

    def hard_update_target_network(self):
        pars = self.model.get_weights()
        self.target_model.set_weights(pars)

    def soft_update_target_network(self):
        pars_behavior = self.model.get_weights()
        pars_target = self.target_model.get_weights()

        ctr = 0
        for par_behavior,par_target in zip(pars_behavior,pars_target):
            par_target = par_target*(1-self.tau) + par_behavior*self.tau
            pars_target[ctr] = par_target
            ctr += 1

        self.target_model.set_weights(pars_target)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def preprocess(self, state):
        return np.reshape(state, [1, self.state_size])

    def act(self, state):
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

            # Choose Action
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            Qs = self.model.predict(state)[0]
            action = np.argmax(Qs)

        return action

    def train(self):
        indices, mini_batch, importance  = self.replay_buffer.sample(self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        Q_wants = []
        td_errors = np.zeros(self.batch_size)

        for i in range(len(mini_batch)):
            if not self.use_PER:
                state, action, reward, next_state, done = mini_batch[i]
            else:
                state = mini_batch[i][0][0]
                action = mini_batch[i][0][1]
                reward = mini_batch[i][0][2]
                next_state = mini_batch[i][0][3]
                done = mini_batch[i][0][4]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states_tensor = np.reshape(states,(self.batch_size,len(states[0])))
        Q_wants_pred = self.model.predict(states_tensor)

        next_states_tensor = np.reshape(next_states,(self.batch_size,len(next_states[0])))
        Q_next_state_vecs = self.model.predict(next_states_tensor)
        Q_target_next_state_vecs = self.target_model.predict(next_states_tensor)

        for i in range(len(mini_batch)):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]

            Q_want = Q_wants_pred[i]
            Q_want_old = Q_want[action]

            if done:
                Q_want[action] = reward
            else:
                Q_next_state_vec = Q_next_state_vecs[i]
                action_max = np.argmax(Q_next_state_vec)

                Q_target_next_state_vec = Q_target_next_state_vecs[i]
                Q_target_next_state_max = Q_target_next_state_vec[action_max]

                Q_want[action] = reward + self.gamma*Q_target_next_state_max
                Q_want_tensor = np.reshape(Q_want,(1,len(Q_want)))

            Q_wants.append(Q_want)
            td_errors[i] = abs(Q_want_old - Q_want[action])

        states = np.array(states)
        Q_wants = np.array(Q_wants)
        self.model.fit(states, Q_wants, verbose=False, epochs=1)

        # update replay buffer
        self.replay_buffer.batch_update(indices, np.array(td_errors))

    def save(self, file='model.h5'):
        print('Save model...')
        self.model.save(file)
