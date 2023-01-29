import numpy as np
import random
import os
import time
import itertools
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, Image
from collections import namedtuple

# using (y, x) order for co-ordinates because that's what numpy uses
# in retrospect I wish I'd just flipped the array first and used (x, y)
CarState = namedtuple('CarState', "y x v_y v_x")

CarAction = namedtuple('CarAction', "a_y a_x")

CarExperience = namedtuple('CarExperience', "state action extra")

def time_func(name, func):
  start = time.time()
  result = func()
  end = time.time()
  took_seconds = int(end - start)
  print(f"[TIMETAKEN] Algo={name}, Duration={took_seconds} seconds")
  return result

def load_track_file(track_path):
        """
        Load a track file from disk into numpy 2d array

        :param track_path: path on disk to find file
        :return: 2d numpy array of track
        """
        with open(track_path) as f:
            lines = f.readlines()

        first_line = lines[0].split(",") # first line contains dims

        track = np.asarray([list(line.strip("\n")) for line in lines[1:]])

        if track.shape[0] != int(first_line[0]):
          raise Exception(f"track had invalid number of rows: {track.shape[0]}")
        if track.shape[1] != int(first_line[1]):
          raise Exception(f"track had invalid number of cols: {track.shape[1]}")

        return track

class Environment(object):
 
  def __init__(self, track, max_velocity=5, max_acceleration=1, crash_result="starting", **kwargs):
    """
      Initialize the enviroment

      :param :track: name of the track
      :param :max_velosity = 5 (maximum velocity allowed)
      :param :max_acceleration = 1
      :crash_result: crashing scenario deciding cars next state

    """
    self.track = track
    self.crash_result = crash_result

    #initialize velocity
    velocity_space = np.arange(max_velocity * -1, max_velocity + 1, 1)

    #initialize the states with x and y positions and x and y velocities
    self.state_space = [
        np.arange(0, self.track.shape[0]), # y_position
        np.arange(0, self.track.shape[1]), # x_position
        velocity_space,                    # y_velocity
        velocity_space                     # x_velocity
    ]

    #determines best acceleration for action
    acceleration_options = list(range(max_acceleration*-1, max_acceleration+1))  

    #initialize action space
    self.action_space = [CarAction(a_x, a_y) for a_x in acceleration_options for a_y in acceleration_options]

    #start positions
    self.start_positions = np.asarray(list(zip(*np.where(track == "S"))))

    #cars finish positions
    self.finish_positions = np.asarray(list(zip(*np.where(track == "F"))))

    #car's initial state
    self.initial_state = CarState(*random.choice(self.start_positions), 0, 0)

  def build_V(self):
    """
    builds state-value tuple

    return: state-value tuple
    """
    return np.random.uniform(size=tuple([len(space) for space in self.state_space]))

  def build_Q(self):
    """
    defines the terminal state/finish line

    return: x and y position at the terminal state
    """
    return np.random.uniform(size=(* tuple([len(space) for space in self.state_space]), len(self.action_space)))

  def is_terminal_state(self, state):
    """
    determines if the car is not on the track and puts it back on the track

    return: new randomly picked car positions that are on the track
    """
    return self.track[state.y, state.x] == "F"

  def enumerate_possible_agent_states(self):
    """
    # Goes through all possible states
    # filters out the states where the car is sittng on a wall
    # because we won't allow the car to ever need to choose an action while on a wall
    # it will automatically get moved to a different (non-wall) state by the crash result policy
    """
    return (CarState(*permutation) 
            for permutation in itertools.product(*self.state_space) 
            if self.track[permutation[0], permutation[1]] != "#")

  def get_action_result(self, state, action, probability_no_action=0.2, print_when_action_failed=False):
    """
    Decides action based on the probability given
    :Param: state (current_state)
    :Param: action (current_action)
    :Param: Probability of no action = 0.2
    
    """
    extra = {}
    # apply acceleration to velocity before moving position
    if np.random.uniform() > probability_no_action:
      # if we get to update accelerations, add acceleration to velocity and have min/max from velocity space
      v_y = min(max(state.v_y + action.a_y, self.state_space[2][0]), self.state_space[2][-1])
      v_x = min(max(state.v_x + action.a_x, self.state_space[3][0]), self.state_space[3][-1])
    else:
#       if DEMONSTRATE_STATE_ACTION_STATE and np.random.uniform() <= probability_no_action:  
#         print("non-deterministic action")
        
      if print_when_action_failed:
        print("action failed, no change to velocity", "x-velocity", state.v_x, "y-velocity", state.v_y )
      # action "failed", don't change velocity
      extra["action_failed"] = True
      v_y = state.v_y
      v_x = state.v_x
    
    # move position based on velocity
    if DEMONSTRATE_NON_DETERMINISM:
        print("Action = Change velocity", "x-velocity", v_x, "y-velocity", v_y )
    y = state.y + v_y
    x = state.x + v_x

    # handle out of bounds or crashes
    if self._is_out_of_bounds(y, x) or self._did_crash(state.y, state.x, v_y, v_x):
      next_state = self._new_state_after_crash(state.y, state.x, v_y, v_x)
    else:
      next_state = CarState(y, x, v_y, v_x)

    # find reward
    if self.track[next_state.y, next_state.x] == "F":
      reward = 0
    else:
      reward = -1

    return next_state, reward, extra

  def _is_out_of_bounds(self, y, x):
    """ 
    determines if the car is out of bounds
    :param: y (y-position on the track)
    :param: x (x-position on the track)
    
    :returns x and y positions of the car when out of bounds
    """
    return y >= self.track.shape[0] or y < 0 or x >= self.track.shape[1] or x < 0

  def _did_crash(self, y, x, v_y, v_x):
    """ 
    determines if the car is crashed on the wall, that is if the car is at "#"
    :param: y (y-position on the track)
    :param: x (x-position on the track)
    :param: v_y (velocity in y direction)
    :param: v_x (velocity in x direction)
    
    :returns x and y positions of the car when crashed
    """
    for position in self._car_step_traveled_positions(y, x, v_y, v_x):
        if self.track[position] == "#":
            if DEMONSTRATE_CRASH_DETECTION:
                print("Car crashed", y, x, v_y, v_x)
            return True
    return False

  def _car_step_traveled_positions(self, y, x, v_y, v_x):
    """
    Get car's travelled positions
    :param: y (y-position on the track)
    :param: x (x-position on the track)
    :param: v_y (velocity in y direction)
    :param: v_x (velocity in x direction)
    """
    fraction_count = max(abs(y), abs(x))

    position = (y, x)
    # yield position

    for step in range(1, fraction_count+1):
        step_fraction = float(step) / float(fraction_count) 
        next_position = (y + int(v_y * step_fraction), x + int(v_x * step_fraction))
        if position != next_position:
            yield next_position
        position = next_position

  def _new_state_after_crash(self, y, x, v_y, v_x):
    """
    Get car's new state after the crash
    :param: y (y-position on the track)
    :param: x (x-position on the track)
    :param: v_y (velocity in y direction)
    :param: v_x (velocity in x direction)
    """
    #new state if the crash penalty puts the car at the starting positions
    if self.crash_result == "starting":
      if DEMONSTRATE_CAR_RESTART:
        print("CrashResult=Starting, new state=",self.initial_state)
      return self.initial_state
    
    #new state if the crash penalty puts the car at the nearest positions
    elif self.crash_result == "nearest":
      position = (y, x)  
      for next_position in self._car_step_traveled_positions(y, x, v_y, v_x):
         if self.track[next_position] == "#":
            if DEMONSTRATE_CAR_RESTART:
                print("CrashResult=Nearest, new state=", CarState(*position, 0, 0))
            return CarState(*position, 0, 0)
         position = next_position
      raise Exception("could not find crash wall")
    raise Exception(f"unsupported: {self.crash_result}")

def test_policy(environment, Q, probability_no_action=0.2, limit_steps=200, debug=False, **kwargs):
# """
# creates state-action sequence tuples of all learned policies
# :param: environment of the state-action
# :param: Q state
# :param: probability_no_action = 0.2
# :param: limit_steps = 200 policies

# returns the state sequence
# """
  state_sequence = []
  state = environment.initial_state
  for step_count in range(1, limit_steps+1):  
    action = environment.action_space[np.argmax(Q[state])]

    next_state, reward, extra = environment.get_action_result(state, action, probability_no_action)
    
    state_sequence.append(CarExperience(state, action, extra))
    
    state = next_state

    #if terminal, leave the track
    if environment.is_terminal_state(state):
        # state_sequence.append(CarExperience(state, CarAction(0, 0)))
        break

  return state_sequence

def test_policy_average(environment, Q, **kwargs):
  test_sequences = [test_policy(environment, Q, **kwargs) for i in range(100)]

  return np.mean([len(test_sequence) for test_sequence in test_sequences])

def show_plot(data, title="chart", x_axis="iterations", y_axis="avg number of steps", font_size=12):
# """
# generate plot of the minimum steps to required to finish the race per iteration
# :param: chart title
# :x-axis: x-axis name
# :y_axis: y-axis name
# :font_size: font size set to 12
# """

  xs, ys = zip(*sorted(data.items()))
  plt.figure(figsize=(20, 5))
  plt.plot(xs, ys)
  plt.ylim([0, max(data.values()) + 30])
  plt.xlabel(x_axis, fontsize=font_size)
  plt.ylabel(y_axis, fontsize=font_size)
  #plt.ticklabel_format(axis="x", style="sci")
  #plt.xaxis.major.formatter._useMathText = True
  
  plt.title(title, fontsize=font_size + 2)

  for x,y in list(zip(xs,ys))[::2]:
    plt.annotate("{:.0f}".format(y), (x,y), textcoords="offset points",
                 xytext=(0,10), ha='center')
  
  plt.show()

def kwargs2title(**kwargs):
  title = ""
  for k, v in sorted(kwargs.items()):
    if k in ["num_iterations"]:
      continue
    title = title + f"{k}={v}, "

  return title

def train_value_iteration(environment, num_iterations, probability_no_action=0.2, discount_factor=0.9, **kwargs):
#     """
#     train value iteration algorithn

#     :param: environment
#     :param: num_iterations: iterations to train the algorithm for
#     :param: probability_no_action = 0.2 
#     :param: discount_factor = 0.9

#     returns Q-values, and average steps over iterations
#     """

  V, Q = (environment.build_V(), environment.build_Q())
  steps_over_iterations = {}

  for iteration in range(num_iterations):
    for state in environment.enumerate_possible_agent_states():
      for action_idx, action in enumerate(environment.action_space):
        
        #new state if there is action
        state_if_action, reward_if_action, _ = environment.get_action_result(state, action, probability_no_action=0)
        expected_value_if_action = V[state_if_action]
        
        #demonstrate state, action, state triplets
        if DEMONSTRATE_STATE_ACTION_STATE and iteration == 2:
            print("\n state", state, "action", action, "new_state", state_if_action)
        
        #no action, state doesn't change
        print_when_action_failed = DEMONSTRATE_NON_DETERMINISM and iteration == 2
        state_if_failed_action, reward_if_failed_action, _ = environment.get_action_result(state, action, probability_no_action=1,print_when_action_failed =print_when_action_failed )
        expected_value_if_failed_action = V[state_if_failed_action]
        
#         if DEMONSTRATE_STATE_ACTION_STATE and iteration == 2:
#             print("\n state", state, "no action", action, "new_state", state_if_failed_action)

        #calculate immediate reward
        immediate_reward = (
          ((1 - probability_no_action) * reward_if_action)
          + (probability_no_action * reward_if_failed_action)
        )

        #calculate expected future reward
        expected_future_value = (
            ((1 - probability_no_action) * expected_value_if_action)
            + (probability_no_action * expected_value_if_failed_action)
        )
        
        #calculate new Q

        Q[(*state, action_idx)] = immediate_reward + (discount_factor * expected_future_value)
      
    
    #update V value
      V[state] = np.max(Q[state])
      if SHOW_V_VALUES_UPDATING and iteration == 2:
        print("iteration", iteration, "V Value", V[state])

    avg_steps = test_policy_average(environment, Q, **kwargs)
    steps_over_iterations[iteration] = avg_steps
    print(".", end = '|' if iteration % 10 == 9 else '')
  
  print(f"\n[DONE] Last Avg Steps = {avg_steps}")

  return Q, steps_over_iterations

def train_q_learning(
    environment,
    num_iterations,
    start_from="random",
    probability_no_action=0.2,
    learning_rate=0.2,
    discount_factor=0.9,
    limit_steps_per_episode=20,
    **kwargs):

#     """
#     train Q-learning algorithm
#     :param: environment
#     :param: num_iterations: iterations to train the algorithm for
#     :param: probability_no_action = 0.2 
#     :param: discount_factor = 0.9

#     returns Q-values, and average steps over iterations
#     """
  
  Q = environment.build_Q()
  steps_over_iterations = {}

  valid_start_states = list(environment.enumerate_possible_agent_states())

  for iteration in range(num_iterations):

    #initialize states for starting from a random position vs the starting position
    if start_from == "random":
      state = random.choice(valid_start_states)
    elif start_from == "start":
      state = environment.initial_state

    for _ in range(limit_steps_per_episode):
        if environment.is_terminal_state(state):
            break

        # pick best action based on best existing expected rewards
        action_idx = np.argmax(Q[state])
        action = environment.action_space[action_idx]

        if PRINT_Q_Q_VALUES and iteration == 10000:
            print("Step Number", _, "Previous Q Values for Q-Learning", Q[(*state, action_idx)])
          # get next state and reward from environment
        next_state, reward, _ = environment.get_action_result(state, action, probability_no_action=probability_no_action)

        #print previous Q values
        if DEMONSTRATE_STATE_ACTION_STATE and iteration == 10000:
            "STARTING Q-value State-action-state printing"
            print("\n state", state, "action", action, "new_state", state_if_action)
            "\n END Q-value State-action-state printing\n"


          # update Q
        Q[(*state, action_idx)] = (
            ((1 - learning_rate) * Q[(*state, action_idx)]) # previous value scaled down
        + (learning_rate * (reward + discount_factor * np.max(Q[next_state])))
        )

        #print updated Q values
        if PRINT_Q_Q_VALUES and iteration == 10000:
            print("Step Number", _, "New Q Values for Q-Learning", Q[(*state, action_idx)])

        state = next_state



    if iteration % 50000 == 0:
      avg_steps = test_policy_average(environment, Q, **kwargs)
      steps_over_iterations[iteration] = avg_steps
      print(".", end = '|' if iteration % 50000 == 499999 else '')

  print(f"\n[DONE] Last Avg Steps = {avg_steps}")

  return Q, steps_over_iterations

def train_sarsa(environment, num_iterations, start_from="random", probability_no_action=0.2, learning_rate=0.2, discount_factor=0.9, limit_steps_per_episode=20, **kwargs):

#     """
#   train Q-learning algorithm
#   :param: environment
#   :param: num_iterations: iterations to train the algorithm for
#   :param: probability_no_action = 0.2 
#   :param: discount_factor = 0.9

#   returns Q-values, and average steps over iterations
  
#     """

  Q = environment.build_Q()
  steps_over_iterations = {}

  valid_start_states = list(environment.enumerate_possible_agent_states())

  for iteration in range(num_iterations):
    
    #initialize states for starting from a random position vs the starting position
    if start_from == "random":
      state = random.choice(valid_start_states)
    elif start_from == "start":
      state = environment.initial_state

    # pick best action based on best existing expected rewards
    action_idx = np.argmax(Q[state])

    for _ in range(limit_steps_per_episode):
        if environment.is_terminal_state(state):
            break
        
        # get next state and reward from environment
        next_state, reward, _ = environment.get_action_result(
            state,
            environment.action_space[action_idx],
            probability_no_action=probability_no_action)
      
        #demonstrate state action state 
        if DEMONSTRATE_STATE_ACTION_STATE and iteration == 10000:
            "STARTING SARSA State-action-state printing"
            print("\n state", state, "action", action, "new_state", state_if_action)
            "\n END SARSA State-action-state printing\n"
        next_action_idx = np.argmax(Q[next_state])

        #demonstrate Q values
        if PRINT_SARSA_Q_VALUES and iteration == 10000:
            print("Step Number", _, "Previous Q Values for SARSA", Q[(*state, action_idx)])
        
        # update Q
        Q[(*state, action_idx)] = (
            ((1 - learning_rate) * Q[(*state, action_idx)]) # previous value scaled down
        + (learning_rate * (reward + discount_factor * Q[(*next_state, next_action_idx)]))
        )
        
        if PRINT_SARSA_Q_VALUES and iteration == 10000:
            print("Step Number", _, "New Q Values for SARSA", Q[(*state, action_idx)])
        
        state = next_state
        action_idx = next_action_idx
        

    if iteration % 50000 == 0:
      avg_steps = test_policy_average(environment, Q, **kwargs)
      steps_over_iterations[iteration] = avg_steps
      print(".", end = '|' if iteration % 50000 == 499999 else '')

  print(f"\n[DONE] Last Avg Steps = {avg_steps}")

  return Q, steps_over_iterations

def run_experiment(**kwargs):
  title = kwargs2title(**kwargs)
  print(f"[START] Training experiment with: {title}")

  algos_dict = {
      "value_iteration": train_value_iteration,
      "q_learning":train_q_learning,
      "sarsa": train_sarsa
  }

  environment = Environment(load_track_file(kwargs["track_path"]), **kwargs)
  
  print("[RUNNING] ", end="")
  Q, steps_over_iterations = time_func(kwargs["algo"], lambda: algos_dict[kwargs["algo"]](environment, **kwargs))

  show_plot(steps_over_iterations, title)

  steps = test_policy(environment, Q)

  #print(steps)

  return Q, steps_over_iterations, environment

# TODO test whether it's better to start from start or a random place for Q and sarsa
# TODO does the car "cheat" by crashing drop velocity around corners

algos = ["value_iteration", "q_learning", "sarsa"]
tracks = ["L-track.txt", "O-track.txt", "R-track.txt"]
crash_results = ["starting", "nearest"]

all_experiments = [{
    "algo": p[0],
    "track_path": p[1],
    "crash_result": p[2],
    "num_iterations": 60 if p[0] == "value_iteration" else 1500000 }
    for p in itertools.product(algos, tracks, crash_results)
]


for experiment_params in all_experiments:
  Q, performance, environment = run_experiment(**experiment_params)



DEMONSTRATE_NON_DETERMINISM = False
SHOW_V_VALUES_UPDATING = False
PRINT_Q_Q_VALUES = False
PRINT_SARSA_Q_VALUES = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_CAR_RESTART = False
DEMONSTRATE_CRASH_DETECTION = False


DEMONSTRATE_NON_DETERMINISM = False
SHOW_V_VALUES_UPDATING = True
PRINT_Q_Q_VALUES = False
PRINT_SARSA_Q_VALUES = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_CAR_RESTART = False
DEMONSTRATE_CRASH_DETECTION = False

print ("starting Value-iteration Algorithm")
Value_Q, value_performance, value_environment = run_experiment(
    algo="value_iteration",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=3)

DEMONSTRATE_NON_DETERMINISM = False
SHOW_V_VALUES_UPDATING = False
PRINT_Q_Q_VALUES = True
PRINT_SARSA_Q_VALUES = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_CAR_RESTART = False
DEMONSTRATE_CRASH_DETECTION = False
print ("starting Q-learning Algorithm")
Q_Q, Q_performance, Q_environment = run_experiment(
    algo="q_learning",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=15000)

DEMONSTRATE_NON_DETERMINISM = False
SHOW_V_VALUES_UPDATING = False
PRINT_Q_Q_VALUES = False
PRINT_SARSA_Q_VALUES = True
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_CAR_RESTART = False
DEMONSTRATE_CRASH_DETECTION = False

print ("starting SARSA Algorithm")
SARSA_Q, SARSA_performance, SARSA_environment = run_experiment(
    algo="sarsa",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=15000)

DEMONSTRATE_NON_DETERMINISM = True
SHOW_V_VALUES_UPDATING = False
PRINT_Q_Q_VALUES = False
PRINT_SARSA_Q_VALUES = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_STATE_ACTION_STATE = False
DEMONSTRATE_CAR_RESTART = False
DEMONSTRATE_CRASH_DETECTION = False

print ("starting Value-iteration Algorithm")
Value_Q, value_performance, value_environment = run_experiment(
    algo="value_iteration",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=3)

print ("starting Q-learning Algorithm")
Q_Q, Q_performance, Q_environment = run_experiment(
    algo="q_learning",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=15000)


print ("starting SARSA Algorithm")
SARSA_Q, SARSA_performance, SARSA_environment = run_experiment(
    algo="sarsa",
    track_path="L-track.txt",
    crash_result="starting",
    num_iterations=15000)


v_test_policy = test_policy(value_environment, Value_Q)
q_test_policy = test_policy(Q_environment, Q_Q)
s_test_policy = test_policy(SARSA_environment, SARSA_Q)
