
import subprocess


def kill_process(pid: int):
    try:
        subprocess.check_output(['kill', '-9', '{:d}'.format(pid)])
        print("Killed process: {:d}".format(pid))
    except:
        print("Failed to kill process: {:d}".format(pid))


if __name__ == '__main__':
    try:
        gpu_status = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'  # , '--id={:d}'.format(0)
        ])
    except subprocess.CalledProcessError:
        print("nvidia-smi failed, no GPU card detected")
        exit()

    process_to_be_killed = gpu_status.decode().splitlines()
    yes_or_no = input(f"Are you sure to kill these process: {process_to_be_killed}: (y/n)")
    if yes_or_no.lower() == 'y':
        for pid in process_to_be_killed:
            kill_process(int(pid))
