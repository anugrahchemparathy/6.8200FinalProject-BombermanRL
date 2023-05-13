from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# read tf log file
def read_tf_log(log_dir):
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f'**/events.*'))
    print(log_files)
    if len(log_files) < 1:
        return None
    log_file = log_files[0]
    event_acc = EventAccumulator(log_file.as_posix())
    event_acc.Reload()
    tags = event_acc.Tags()
    print(tags)
    # scalar_success = event_acc.Scalars('train/episode_success')
    scalar_return = event_acc.Scalars('train/episode_return/mean')
    # success_rate = [x.value for x in scalar_success]
    steps = [x.step for x in scalar_return]
    returns = [x.value for x in scalar_return]
    return steps, returns

if __name__ == '__main__':
    save_dir = "data/ppo_base"
    steps, returns = read_tf_log(save_dir)
    plt.plot(steps, returns)
    plt.title('Average return for PPO agent')
    plt.xlabel('# steps')
    plt.ylabel('return')
    plt.show()
