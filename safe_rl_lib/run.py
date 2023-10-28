import os
import subprocess
import re
import time
import os.path as osp

def get_gpu_memory(device_id):
    try:
        cmd = f'nvidia-smi --id={device_id} --query-gpu=memory.free --format=csv,noheader'
        result = subprocess.check_output(cmd, shell=True)
        memory_free = int(re.findall(r'\d+', result.decode())[0])
        return memory_free
    except Exception as e:
        print(f"Error getting GPU memory info for device {device_id}: {str(e)}")
        return 0

def get_all_gpu_memory():
    try:
        cmd = 'nvidia-smi --query-gpu=count --format=csv,noheader'
        result = subprocess.check_output(cmd, shell=True)
        gpu_count = int(re.findall(r'\d+', result.decode())[0])
        gpu_memory = [get_gpu_memory(i) for i in range(gpu_count)]
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory info: {str(e)}")
        return []
    
def run(python_files_and_args, available_devices):
    indexed_python_files_and_args = [(index, tup) for index, tup in enumerate(python_files_and_args)]
    
    gpu_not_enough = False
    processes = []
    for index, (file_path, arguments) in enumerate(python_files_and_args):
        gpu_memory = get_all_gpu_memory()
        gpu_memory = [gpu_memory[i] for i in available_devices]
        assert gpu_memory != []
        if max(gpu_memory) < 1300:
            gpu_not_enough = True
            break
        
        best_gpu = available_devices[gpu_memory.index(max(gpu_memory))]
        # vctrpo.py will use the third gpu set by os.environ
        devices = str(best_gpu)
        
        try:
            processes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={devices} python {file_path} {arguments}", shell=True, 
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            print(f"Task {index} successfully executed on cuda:{best_gpu} with params [{file_path} {arguments}]")
        except Exception as e:
            print(f"Error when starting process {index}: {str(e)}")
        time.sleep(10)
    
    print("----------------------------")

    if not gpu_not_enough:
        for process, (index, (file_path, arguments)) in zip(processes, indexed_python_files_and_args):
            if process is not None:
                process.wait()
                if process.returncode == 0:
                    print(f"Task {index} executed successfully with arguments: [{arguments}]")
                else:
                    print(f"Task {index} encountered an error with arguments: [{arguments}]")
    else:
        print("GPU memory is nou enough. Please release the memory manually!")
        # for index, (file_path, arguments) in enumerate(python_files_and_args):
        #     exit_code = os.system(f"pkill -f {arguments}")
        #     if exit_code == 0 :
        #         print(f"Task {index} successfully killed.")
        #     else:
        #         print(f"Kill task {index} failed. Error[{exit_code}].")        

if __name__ == "__main__":
    python_files_and_args = []
            
    for task in ['Goal_Point_8Hazards']:
        for L_beta in [1, 0.5, 0.1, 0.05, 0.01]:
            for dynamic_epochs in [50, 100]:
                python_files_and_args.append(("uaissa/uaissa.py", f"--task {task} --L_beta {L_beta} --dynamic_epochs {dynamic_epochs}"))

    run(python_files_and_args, [0,1,2,3,4,5,6,7])
