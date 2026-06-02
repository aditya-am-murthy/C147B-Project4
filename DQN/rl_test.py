import torch
import torch.nn as nn
import traceback
from termcolor import colored
import gymnasium as gym
import numpy as np
import sys
import pickle

# get the operating system
if sys.platform.startswith("darwin"):
    # Mac OS X
    suffix = "_mac"
else:
    suffix = ""


def test_model_DQN(model):
    try:
        model = model((4, 84, 84), 5)

        model.load_state_dict(torch.load("test_weights.pt", map_location="cpu", weights_only=False))
        # Test the forward function
        test_outputs = torch.load(f"test_outputs{suffix}.pt", map_location=torch.device("cpu"), weights_only=False)
        test_inputs = test_outputs["S"]
        test_outputs = test_outputs["outputs"]
        model.eval()
        with torch.no_grad():
            for i in range(len(test_inputs)):
                # print(torch.tensor(test_inputs[i]).float().shape)
                assert torch.allclose(
                    model(torch.tensor(test_inputs[i]).float().unsqueeze(0)), torch.tensor(test_outputs[i])
                ), f"expected {test_outputs[i]} but got {model(torch.tensor(test_inputs[i]).float().unsqueeze(0))}"

        print(colored("Passed", "green"))
    except Exception as e:
        print(e)
        print(colored("Failed", "red"))
        traceback.print_exc()
        return


# def test_model_DDPG(model):
#     pass


def test_wrapper(wrapper):
    try:
        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        wrapper = wrapper(env)
        PIX_TOL = 0.05  # a pixel differing by more than this counts as a real mismatch
        MAX_BAD_PIX = 100  # tolerate a few edge pixels from cross-version render noise; real bugs hit thousands

        # Test the reset function
        test_outputs = torch.load(f"test_outputs_v3{suffix}.pt", map_location=torch.device("cpu"), weights_only=False)
        test_inputs = test_outputs["outputs"]
        test_outputs = test_outputs["S"]
        s, _ = wrapper.reset(seed=42)

        diff = np.abs(test_outputs[0].astype(np.float64) - np.asarray(s, dtype=np.float64))
        bad = int((diff > PIX_TOL).sum())
        assert bad <= MAX_BAD_PIX, (
            f"reset mismatch: {bad} px over {PIX_TOL} " f"(max diff {diff.max():.4f}, mean diff {diff.mean():.6f})"
        )
        print(colored("Passed reset", "green"))

        for i in range(len(test_outputs) - 1):
            # Test the step function
            s, r, terminated, truncated, info = wrapper.step(test_inputs[i])
            diff = np.abs(test_outputs[i + 1].astype(np.float64) - np.asarray(s, dtype=np.float64))
            bad = int((diff > PIX_TOL).sum())
            assert bad <= MAX_BAD_PIX, (
                f"step {i} mismatch: {bad} px over {PIX_TOL} "
                f"(max diff {diff.max():.4f}, mean diff {diff.mean():.6f})"
            )

        print(colored("Passed step", "green"))
    except Exception as e:
        print(e)
        print(colored("Failed", "red"))
        traceback.print_exc()
        return


def check_same_torch(a, b):
    # first check shape
    if a.shape != b.shape:
        return False
    # then check values
    return torch.allclose(a, b)


def test_DQN_replay_buffer(buffer_class):
    with open(f"test_replay_buffer_inputs{suffix}.pkl", "rb") as f:
        buffer_inputs = pickle.load(f)
    buffer_samples = torch.load(f"test_replay_buffer_samples{suffix}.pth", weights_only=False)
    try:
        buffer = buffer_class(40, seed=42)
        j = 0
        for i in range(100):
            buffer.add(
                buffer_inputs["states"][i],
                buffer_inputs["actions"][i],
                buffer_inputs["rewards"][i],
                buffer_inputs["next_states"][i],
                buffer_inputs["dones"][i],
            )
            if i % 30 == 29:
                # print(i)
                target_outputs = buffer_samples[j]
                actual_outputs = buffer.sample(5)
                for k in range(len(target_outputs)):
                    # print(target_outputs[k],actual_outputs[k])
                    assert check_same_torch(
                        target_outputs[k], actual_outputs[k]
                    ), f"expected {target_outputs[k][0]} but got {buffer.sample(1)[k][0]}"
                # assert np.all(buffer_samples[j] == buffer.sample(40)), f"expected {buffer_samples[j]} but got {buffer.sample(40)}"
                j += 1

        print(colored("Passed", "green"))
    except:
        print(colored("Failed", "red"))
        traceback.print_exc()
        return


if __name__ == "__main__":
    from Project4.DQN.replay_buffer import ReplayBufferDQN

    test_DQN_replay_buffer(ReplayBufferDQN)

    from Project4.DQN.env_wrapper import EnvWrapper

    test_wrapper(EnvWrapper)

    from Project4.DQN.model import Nature_Paper_Conv

    test_model_DQN(Nature_Paper_Conv)
