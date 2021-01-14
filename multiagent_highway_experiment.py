"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import RLController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
    VehicleParams, SumoParams, \
    SumoCarFollowingParams, SumoLaneChangeParams
# from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayRampsNetwork
from flow.envs.multiagent import MultiAgentHighwayPOEnv
from flow.envs.multiagent.highway import ADDITIONAL_ENV_PARAMS
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

#from flow.core.params import SimParams

import json
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from flow.utils.rllib import FlowParamsEncoder

import datetime

name = "highway-ramp-RL"
# SET UP PARAMETERS FOR THE SIMULATION

training_inputs = {
    "N_TRAINING_ITERATIONS": 1000,
    "N_ROLLOUTS": 20,
    "HORIZON": 1500,
    "N_CPUS": 11,
    "HIGHWAY_INFLOW_RATE": 3000,
    "ON_RAMPS_INFLOW_RATE": 350,
    "PENETRATION_RATE": 5.0
}


# SET UP PARAMETERS FOR THE NETWORK
additional_net_params_dictionary = {
    #lengths
    "highway_length":1500,
    "on_ramps_length":250,
    "off_ramps_length":250, 
    #lanes
    "highway_lanes":3, 
    "on_ramps_lanes":1,
    "off_ramps_lanes":1,
    #speed limits
    "highway_speed":30, 
    "on_ramps_speed":20,
    "off_ramps_speed":20,
    #ramps
    "on_ramps_pos":[500],
    "off_ramps_pos":[1000],
    # probability of exiting at the next off-ramp
    "next_off_ramp_proba":.25
}
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update(additional_net_params_dictionary)


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 1,
    'max_decel': 1,
    'target_velocity': 30
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

#Speed mode and lange change mode inputs
speed_mode_input = "all_checks" #collisions
lane_change_mode_input = "sumo_default" # Default

# human vehicles
vehicles.add(
    veh_id="idm",
    car_following_params=SumoCarFollowingParams(
        speed_mode=speed_mode_input, 
        tau=1.5  # larger distance between cars is safer
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=lane_change_mode_input))

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=int(training_inputs["HIGHWAY_INFLOW_RATE"]
                      * (1-training_inputs["PENETRATION_RATE"] / 100)),
    depart_lane="free",
    depart_speed="max",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(training_inputs["HIGHWAY_INFLOW_RATE"]
                      * (training_inputs["PENETRATION_RATE"] / 100)),
    depart_lane="free",
    depart_speed="max",
    name="rl_highway_inflow",
    route="routehighway_0_0")

# add human vehicles on all the on-ramps
for i in range(len(additional_net_params['on_ramps_pos'])):
    inflows.add(
        veh_type="idm",
        edge="on_ramp_{}".format(i),
        vehs_per_hour=int(training_inputs["ON_RAMPS_INFLOW_RATE"]
                      * (1-training_inputs["PENETRATION_RATE"] / 100)),
        depart_lane="free",
        depart_speed="max",
        name="idm_on_ramp_inflow")
    inflows.add(
        veh_type="rl",
        edge="on_ramp_{}".format(i),
        vehs_per_hour=int(training_inputs["ON_RAMPS_INFLOW_RATE"]
                      * (training_inputs["PENETRATION_RATE"] / 100)),
        depart_lane="free",
        depart_speed="max",
        name="av_on_ramp_inflow")


# SET UP FLOW PARAMETERS
flow_param_inputs = {
    "horizon": 3600,
    "warmup_steps":0,
    "sim_step":.2
}
flow_params = dict(
    # name of the experiment
    exp_tag=name+str(training_inputs["PENETRATION_RATE"]),

    # name of the flow environment the experiment is running on
    env_name=MultiAgentHighwayPOEnv,

    # name of the network class the experiment is running on
    network=HighwayRampsNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=flow_param_inputs["horizon"],
        warmup_steps=flow_param_inputs["warmup_steps"],
        sims_per_step=1,  # do not put more than one
        additional_params=additional_env_params,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=flow_param_inputs["sim_step"],
        emission_path="/home/gridsan/alandler/data/",
        render=False,
        restart_instance=True
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


# SET UP RLLIB MULTI-AGENT FEATURES

create_env, env_name = make_create_env(params=flow_params, version=0)

# register as rllib env
register_env(env_name, create_env)

# multiagent configuration
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


POLICY_GRAPHS = {'av': (PPOTFPolicy, obs_space, act_space, {})}

POLICIES_TO_TRAIN = ['av']


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


## CREATE EXPERIMENT

ray.init(num_cpus=training_inputs["N_CPUS"], temp_dir="/home/gridsan/alandler")

# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
#      "class registered in the tune registry.")
alg_run = "PPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = training_inputs["N_CPUS"] - 1  # number of parallel workers
config["train_batch_size"] = training_inputs["HORIZON"] * training_inputs["N_ROLLOUTS"]  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = training_inputs["HORIZON"]  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)




## RUN EXPERIMENTS

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        #"restore": "/ray_results/multiagent_highway/PPO_MultiAgentHighwayPOEnv-v0_0_2020-11-02_03-14-23i1b933dz/checkpoint_276/checkpoint-276",
        "checkpoint_freq": 50,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
#        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": training_inputs["N_TRAINING_ITERATIONS"],  # number of iterations to stop after
        },
    },
})




def recordParamsToText():

    #Write to file entitled name and date, time
    now = datetime.datetime.now()
    date = now.strftime("%m-%d-%y")
    time = now.strftime("%H.%M.%S")

    f = open(name+"_"+date+"_"+time+".txt", "w")
    f.write(name+"_"+date+"_"+time+".txt"+"\n\n")

    #Extract the inputed net params, which are most likely to change.
    for key in additional_net_params_dictionary:
        f.write("data/"+key + ": " + str(additional_net_params[key]) +"\n")

    #Write flow rates, speed/lane modes
    f.write("Highway inflow: " + str(training_inputs["HIGHWAY_INFLOW_RATE"]) + "\n")
    f.write("On ramp inflow: " + str(training_inputs["ON_RAMPS_INFLOW_RATE"]) + "\n")
    f.write("Speed mode: " + str(speed_mode_input) + "\n")
    f.write("Lane change mode: " + str(lane_change_mode_input) + "\n")

    #Write sumo and flow params
    for key in flow_param_inputs:
        f.write(key + ": " + str(flow_param_inputs[key]) +"\n")

    #Write training parameters
    for key in training_inputs:
        f.write(key + ": " + str(training_inputs[key]) +"\n")

recordParamsToText()
