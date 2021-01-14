"""Example of a highway section network with on/off ramps."""
import datetime

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import InFlows, VehicleParams, TrafficLightParams
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayRampsNetwork
from flow.core.experiment import Experiment

def highway_noRL_experiment(length, lanes, speed_lim, H_IN, ON_IN, speed_mode, lane_mode):
    name = 'highway-ramp-noRL'

    # SET UP PARAMETERS FOR THE NETWORK

    additional_net_params_dictionary = {
        #lengths
        "highway_length":length,
        "on_ramps_length":250,
        "off_ramps_length":250, 
        #lanes
        "highway_lanes":lanes, 
        "on_ramps_lanes":1,
        "off_ramps_lanes":1,
        #speed limits
        "highway_speed":speed_lim, 
        "on_ramps_speed":speed_lim*.7,
        "off_ramps_speed":speed_lim*.7,
        #ramps
        "on_ramps_pos":[int(length/2)-250],
        "off_ramps_pos":[int(length/2)+250],
        # probability of exiting at the next off-ramp
        "next_off_ramp_proba":.25
    }

    #Add user inputs to additional_net_params
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params.update(additional_net_params_dictionary)

    # SET UP PARAMETERS FOR THE ENVIRONMENT

    additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
    additional_env_params.update({
        'max_accel': 1,
        'max_decel': 1,
        'target_velocity': speed_lim
    })

    # CREATE VEHICLE TYPES AND INFLOWS

    # inflow rates in vehs/hour
    HIGHWAY_INFLOW_RATE = H_IN
    ON_RAMPS_INFLOW_RATE = ON_IN

    #Speed mode and lange change mode inputs
    speed_mode_input = speed_mode # Safe or aggressive
    lane_change_mode_input = lane_mode # Safe or aggressive

    vehicles = VehicleParams()
    # human vehicles
    vehicles.add(
        veh_id="human",
        car_following_params=SumoCarFollowingParams(
            speed_mode= speed_mode_input, 
            tau=1.5,  # larger distance between cars
        ),
        lane_change_params=SumoLaneChangeParams(lane_change_mode=lane_change_mode_input)
    )

    inflows = InFlows()
    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=HIGHWAY_INFLOW_RATE,
        depart_lane="free",
        depart_speed="max",
        name="highway_flow")
    for i in range(len(additional_net_params["on_ramps_pos"])):
        inflows.add(
            veh_type="human",
            edge="on_ramp_{}".format(i),
            vehs_per_hour=ON_RAMPS_INFLOW_RATE,
            depart_lane="first",
            depart_speed="max",
            name="on_ramp_flow")

    flow_param_inputs = {
        "horizon": 3600,
        "sims_per_step": 5,
        "warmup_steps":0,
        "sim_step":.2
    }
    flow_params = dict(
        # name of the experiment
        exp_tag=name,

        # name of the flow environment the experiment is running on
        env_name=AccelEnv,

        # name of the network class the experiment is running on
        network=HighwayRampsNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            render=True,
            emission_path="./data/",
            sim_step=flow_param_inputs["sim_step"],
            restart_instance=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            additional_params=ADDITIONAL_ENV_PARAMS,
            horizon=flow_param_inputs["horizon"],  # 3600
            sims_per_step=flow_param_inputs["sims_per_step"],  # 5
            warmup_steps=flow_param_inputs["warmup_steps"]
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

        # traffic lights to be introduced to specific nodes (see
        # flow.core.params.TrafficLightParams)
        tls=TrafficLightParams(),
    )

    exp = Experiment(flow_params)
    # run the sumo simulation
    _ = exp.run(1, convert_to_csv=True)


    def recordParamsToText():

        #Write to file entitled name and date, time
        now = datetime.datetime.now()
        date = now.strftime("%m-%d-%y")
        time = now.strftime("%H.%M.%S")

        f = open("data/"+name+"_"+date+"_"+time+".txt", "w")
        f.write(name+"_"+date+"_"+time+".txt"+"\n\n")

        #Extract the inputed net params, which are most likely to change.
        for key in additional_net_params_dictionary:
            f.write("data/"+key + ": " + str(additional_net_params[key]) +"\n")

        #Write flow rates, speed/lane modes
        f.write("Highway inflow: " + str(HIGHWAY_INFLOW_RATE) + "\n")
        f.write("On ramp inflow: " + str(ON_RAMPS_INFLOW_RATE) + "\n")
        f.write("Speed mode: " + str(speed_mode_input) + "\n")
        f.write("Lane change mode: " + str(lane_change_mode_input) + "\n")

        #Write sumo and flow params
        for key in flow_param_inputs:
            f.write(key + ": " + str(flow_param_inputs[key]) +"\n")
            
    recordParamsToText()


speed_lims = [30,40,50,60]
H_INs = [3500,4000,4500]
ON_INs = [400,450,500]
speed_modes = ["obey_safe_speed", "aggressive"]
lane_modes = ["sumo_default","no_right_drive_aggressive"]

for i in range(0,10):
    highway_noRL_experiment(1500,3,30,4000,450,"obey_safe_speed","sumo_default")


