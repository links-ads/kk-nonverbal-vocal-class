import os
import shutil

from pathlib import Path
from transformers import TrainerCallback

class ConfusionMatrixCallback(TrainerCallback):
    """
    Callback class for saving confusion matrix during evaluation.

    This callback moves the generated confusion matrix image to a specific directory based on the model, dataset, and
    finetune method used. The image is saved with a filename that includes the current epoch number.

    Args:
        TrainerCallback: The base class for trainer callbacks.

    """

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Extract the model, dataset, and finetune method from the run name
        run_name = args.run_name
        output_model_name = run_name.split('[KK-PEFT]')[-1]
        model = output_model_name.split('_')[0]
        dataset = output_model_name.split('_')[1]
        finetune_method = output_model_name.split('_')[2]
        dataset_eval = list(metrics.keys())[0].split('_')[1]

        # Create the directory if it does not exist
        cwd = os.getcwd()
        Path(f'{cwd}/images/conf_matrices/{finetune_method}/{dataset}').mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(
                'confusion_matrix.png', 
                f'{cwd}/images/conf_matrices/{finetune_method}/{dataset}/{model}_{state.epoch}_{dataset_eval}.png'
            )
        except:
            print('Error: Could not move confusion matrix to image folder.')

        return