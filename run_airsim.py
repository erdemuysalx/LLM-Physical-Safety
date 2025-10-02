import os
import subprocess
import threading
from collections import defaultdict

from airsim_wrapper import *
import time

import utils

count = defaultdict(int)  # count violation and error


def run_code(code_list, exe_path):
    global count
    # Execute AirSim
    command = f'{exe_path} -windowed -ResX=1280 -ResY=720 -ExecCmds="DisableAllScreenMessages"'

    # subprocess.call(command, shell=True)
    air_sim_process = subprocess.Popen(command, shell=True)
    print("Executing AirSim...")
    time.sleep(5)
    air_sim_process.communicate()
    print("Execute AirSim successfully.")

    print(f"Initializing AirSim...")
    aw = AirSimWrapper()
    print(f"Done.")

    for code in code_list:
        if code == "":
            continue
        try:
            exec(code)
            # count violation
            count["collision"] += 1 if utils.check_collision(aw) is True else 0
            count["nfz"] += 1 if utils.check_nfx(aw) is True else 0
            count["height"] += 1 if utils.check_height(aw) is True else 0
            # reset drone position
            aw.reset()

        except Exception as e:
            print(f"Error: {e}")
            count["error"] += 1
            air_sim_process.kill()
            os.system("taskkill /im Blocks.exe /f")
            print("Restarting AirSim...")
            time.sleep(5)
            air_sim_process = subprocess.Popen(command, shell=True)
            print("AirSim restarted.")
            continue
    return count


if __name__ == "__main__":
    # Start a separate thread for running user code
    code_list = [""]  # code to be executed
    exe_path = r'C:\Users\student\Projects\AirSim\AirSimInspection.exe'  # AirSim.exe file location

    # run simulation to evaluate result
    code_thread = threading.Thread(target=run_code, args=(code_list, exe_path))
    code_thread.start()

    # wait result
    code_thread.join()
    print('Done!!!')

    # save result to somewhere
    print(count)
