
import os
from pathlib import Path
import time
import datetime


def exists(file):
    def check():
        return Path(file).exists()
    return check


def wait_until(check_func, check_interval, *fargs, **fkwargs):
    while not check_func(*fargs, **fkwargs):
        print(f"Not triggered in {datetime.datetime.now()}")
        time.sleep(check_interval)


if __name__ == '__main__':
    wait_until(exists("checkpoint.pth"), 10)

    print(f"Triggered in {datetime.datetime.now()}...")
    time.sleep(60)

    # command = "echo 'Hello World...'"
    command = "zsh scripts/train.sh"
    print(f'Launch the command: "{command}" in {datetime.datetime.now()}...')

    os.system(command)
