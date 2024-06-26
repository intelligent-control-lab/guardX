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
        if max(gpu_memory) < 3200:
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
        if "plot" not in file_path:
            time.sleep(10)
    
    print("----------------------------")

    if not gpu_not_enough:
        for process, (index, (file_path, arguments)) in zip(processes, indexed_python_files_and_args):
            if process is not None:
                process.wait()
                if process.returncode == 0:
                    print(f"Task {index} executed successfully with arguments: [{file_path}, {arguments}]")
                else:
                    print(f"Task {index} encountered an error with arguments: [{file_path}, {arguments}]")
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
    # 'a2c', 'apo', 'trpo', 'ppo'
    # 'cpo', 'pcpo', 'pdo', 'scpo', 'trpofac', 'trpoipo', 'trpolag',    'usl', 'lpg', 'safelayer'
    # 有target_kl的: 'cpo', 'pcpo', 'pdo', 'scpo', 'trpofac', 'trpoipo', 'trpolag'
     
    # 结果合理的：cpo, pcpo, trpofac, trpolag, safelayer, trpoipo, usl
    # 结果不合理的：pdo(实现问题)，lpg(实现问题)，'scpo'
    # for algo in ['cpo', 'pcpo', 'trpofac', 'trpolag', 'safelayer', 'trpoipo', 'usl', 'trpo']: 
    #     for seed in [0,1]:
    #         python_files_and_args.append((f"{algo}_one_episode/{algo}.py", f"--task Goal_Walker_8Hazards --seed {seed} --exp_name {algo}_oe --epochs 150 --env_num 4000"))
    
    # for algo in ['a2c', 'ppo', 'trpo', 'apo', 'alphappo', 'vmpo', 'espo']: 
    #     for seed in [0,1]:
    #         python_files_and_args.append((f"{algo}_one_episode/{algo}.py", f"--task Goal_Walker_8Hazards --seed {seed} --epochs 100 --env_num 600"))
    
    # for robo in ['Point', 'Ant', 'Swimmer', 'Walker']: 
    #     for seed in [0,1]:
    #         python_files_and_args.append((f"papo_one_episode/papo.py", f"--task Goal_{robo}_8Hazards --seed {seed} --exp_name papo1_oe --epochs 100 --env_num 600 --omega1 0.001 --omega2 0.005"))
    #         python_files_and_args.append((f"papo_one_episode/papo.py", f"--task Goal_{robo}_8Hazards --seed {seed} --exp_name papo2_oe --epochs 100 --env_num 600 --omega1 0.005 --omega2 0.01"))
    #         python_files_and_args.append((f"papo_one_episode/papo.py", f"--task Goal_{robo}_8Hazards --seed {seed} --exp_name papo3_oe --epochs 100 --env_num 600 --omega1 0.01 --omega2 0.01"))
            
    log_dict = {}
    base_dir = "../Final_Results"
    origin_dirs = os.listdir(base_dir)
    for origin_dir in origin_dirs:
        log_dict[origin_dir] = [osp.join(base_dir, origin_dir, single_dir) for single_dir in os.listdir(osp.join(base_dir, origin_dir))]
    
    for key in log_dict.keys():
        for dire in log_dict[key]:
            python_files_and_args.append(("utils/plot_all.py", f"{dire}/ -s 28 --title {key}/{dire.split('/')[-1]} --value MaxEpLenRet --cost --reward"))
    
    
    run(python_files_and_args, [0,1,2,3])