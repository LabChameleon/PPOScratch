import multiprocessing as mp
import numpy as np
import cloudpickle

from gym import make as gym_make

# Consider stable-baseline3 multiprocessing environments

class CloudpickleWrapper:
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)

def _gym_worker(
        remote: mp.connection.Connection,
        parent_remote: mp.connection.Connection,
        env_fn_wrapper: CloudpickleWrapper,
):
    parent_remote.close()
    env = env_fn_wrapper.var
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class MultProcEnvGym:
    def __init__(self, env_name, num_envs, env_kwargs={}):
        self.closed = False
        ctx = mp.get_context("forkserver")
        self.remotes, self.work_remotes =  zip(*[ctx.Pipe() for _ in range(num_envs)]) # debug what does this mean??
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            env = gym_make(env_name, **env_kwargs)
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(env),
            )
            process = ctx.Process(target=_gym_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def seed(self, seed):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed(True)
