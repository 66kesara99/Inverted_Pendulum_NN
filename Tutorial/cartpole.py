import gym
import random
import numpy as np
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3 #learning rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200 #maximum score
score_requirement = 50 #threshold for good score
initial_games = 10000 #maximum games for random data set

#play a random game
def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


#some_random_games_first()


#create a random population to get some good moves
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    # iterating over games
    for _ in range(initial_games):
        #play the game
        
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):

            #action is 0 or 1
            action = random.randrange(0,2)

            #run the cartpole step by step
            #game data, is balanced or not, game over or not, info
            observation, reward, done, info = env.step(action)

            #save previous observation and the next action
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation

            #if balanced add score
            score += reward

            #end if balance lost
            if done:
                break

        #score is more than accepted, save the score
        if score >= score_requirement:
            accepted_scores.append(score)

            #record the action,  left or right
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]

                elif data[1] == 0:
                    output = [1, 0]

                #add good moves into the training data list
                training_data.append([data[0], output])

        # end of game
        env.reset()
        scores.append(score)

    
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print ('Average accepted score:', mean(accepted_scores))
    print ('Median accepted score:', median(accepted_scores))
    print (Counter(accepted_scores))

    return training_data


#Create the model
def neural_network_model(input_size):
    #input layer
    network = input_data(shape=[None, input_size, 1], name='input')

    #layer nodes=128, activation relu
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    #output layer, output nodes=2, activation softmax
    network = fully_connected(network, 2, activation='softmax')

    #optimization
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    #create the model
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


#Train model
def train_model(training_data, model=False):

    #training data is reshaped
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)

    #labels
    y = [i[1] for i in training_data]

    #create the model
    if not model:
        model = neural_network_model(input_size = len(X[0]))

    #train the model
    model.fit(X, y, n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openaistuff')

    return model


#run the program
training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

#Play the game and get the average results
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        #display playing the game
        env.render()

        #first take random step
        if len(prev_obs) == 0:
            action = random.randrange(0,2)

        #then play by predicted results by the model
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        #record the actions
        choices.append(action)

        #balance the pole
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation

        #record actions (can be used later)
        game_memory.append([new_observation, action])

        #calculate score and stop if lost balance
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score', sum(scores)/len(scores))

#print action counts
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),
                                          choices.count(0)/len(choices)))
