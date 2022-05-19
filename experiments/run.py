import sys
import os
import copy
import json
import time

# nohup python run.py > 2022_01_31_run.out 2> 2022_01_31_run.err &
def get_command(scenario_index = 8, exp_index = 0, cuda_index = 1, alg = "md"):
    scenario_list = ["simple","simple_reference", "simple_speaker_listener", "simple_spread",
                        "simple_adversary", "simple_crypto", "simple_push",
                        "simple_tag", "simple_world_comm"]
    num_adv_list = [0, 0, 0, 0, 1, 1, 1, 3, 4]
    exp_name_list = ["e0" + str(i) for i in range(1,10,1)] + ["e" + str(i) for i in range(10,21,1)]
    exp_name = alg + "_s" + str(scenario_index) + "_" + str(exp_name_list[exp_index])
    exp_path = " > results/s" + str(scenario_index) + "/" + exp_name + ".out 2> results/s" \
                    + str(scenario_index) + "/" + exp_name + ".err &"

    opt1 = dict()
    opt1['scenario'] = scenario_list[scenario_index]
    opt1['num-adversaries'] = num_adv_list[scenario_index]
    opt1['save-dir'] = "models/s" + str(scenario_index) + "/" + exp_name + "/"
    opt1['adv-dir'] = "../../MyAlgorithm/v3-rmaddpg-master/experiments/models/s" + str(scenario_index) + "/" + "ma_s" + str(scenario_index) + "_" + str(exp_name_list[exp_index]) + "/"
    opt2 = dict()
    opt2['exp-name'] = exp_name + exp_path
    

    def generate_command(opt1, opt2, cuda_index):
        cmd = 'CUDA_VISIBLE_DEVICES=' + str(cuda_index) + ' nohup python train_with_adv.py'
        for opt, val in opt1.items():
            cmd += ' --' + opt + ' ' + str(val)
        for opt, val in opt2.items():
            cmd += ' --' + opt + ' ' + str(val)
        return cmd
    
    return generate_command(opt1, opt2, cuda_index)

def run_index(scenario_index, cuda_index, exp_index):
    opt = get_command(scenario_index = scenario_index, exp_index = exp_index, cuda_index = cuda_index, alg = "md")
    print(opt)
    os.system(opt)
    time.sleep(10) # sleep for 1h

def run(scenario_index, cuda_index):
    #run_index(scenario_index, cuda_index, 0)
    #run_index(scenario_index, cuda_index, 1)
    run_index(scenario_index, cuda_index, 2)
    print("------------------------sleep------------------------")
    #    time.sleep(3600) # sleep for 1h
#     will take 5 h
# 24h can finish about 5 scenarios

# take 5 * 8 = 40 hour
run(scenario_index = 1, cuda_index = 2)
#run(scenario_index = 2)
run(scenario_index = 3, cuda_index = 2)
run(scenario_index = 4, cuda_index = 3)
#run(scenario_index = 5)
#run(scenario_index = 6)
run(scenario_index = 7, cuda_index = 3)
#run(scenario_index = 8)