import shutil
import os
import sys

def get_latest_checkpoint(checkpoints):
    if not os.path.isdir(checkpoints):
        os.makedirs(checkpoints)
        return None

    checkpoints = [os.path.join(checkpoints, name) for name in os.listdir(checkpoints)]
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    else:
        return None

def get_best_checkpoint(checkpoints):
    checkpoints = [os.path.join(checkpoints, name) for name in os.listdir(checkpoints)]

    if not checkpoints:
        raise Exception("Couldn't find any weights in the checkpoints/ directory.")

    lowest_loss = sys.float_info.max
    best_checkpoint = checkpoints[0]

    for checkpoint in checkpoints:
        loss = float(checkpoint.split("-")[3])

        if loss < lowest_loss:
            lowest_loss = loss
            best_checkpoint = checkpoint

    print(f"*** Found checkpoint with the best weights: {best_checkpoint} ***")
    return best_checkpoint

def clean_data_and_checkpoints(rundata, checkpoints):
    shutil.rmtree(rundata, ignore_errors=True)
    shutil.rmtree(checkpoints, ignore_errors=True)