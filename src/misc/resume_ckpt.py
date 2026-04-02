import os

from collections import OrderedDict


# Function to extract the step number from the filename
def extract_step(file_name):
    step_str = file_name.split("-")[1].split("_")[1].replace(".ckpt", "")
    return int(step_str)


def find_latest_ckpt(ckpt_dir):
    # List all files in the directory that end with .ckpt
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    # Check if there are any .ckpt files in the directory
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in {ckpt_dir}.")
    else:
        # Find the file with the maximum step
        latest_ckpt_file = max(ckpt_files, key=extract_step)

        return ckpt_dir / latest_ckpt_file


def no_resume_upsampler(pretrained_state_dict):
    new_state_dict = OrderedDict()
    for key, value in pretrained_state_dict.items():
        if 'upsampler' not in key:
            new_state_dict[key] = value

    return new_state_dict


def load_partial_state_dict(model, pretrained_state_dict):
    # Load only matching parameters
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    # for key in model_state_dict:
    #     if key not in filtered_state_dict:
    #         print(key)
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
