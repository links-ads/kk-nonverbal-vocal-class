import os
import csv
import argparse
from .configs import ModelConfig
from .models import ModelsFactory
from .preprocessing import (
    PreprocessorConfig,
    get_label_weights
)
from .training import (
    CollatorFactory,
    compute_metrics,
    load_dataset,
)
from transformers import (
    TrainingArguments,
    Trainer, 
    PretrainedConfig
)


def main(model_config_path: str, training_config_path: str, preprocessing_config_path: str) -> None:        
    """
        Main function to train a model with the specified parameters.

        Args:
            model_config_path (str): Path to the model configuration file.
            training_config_path (str): Path to the training configuration file.
            preprocessing_config_path (str): Path to the preprocessing configuration file.

        Returns:
            None
    """
    # Load configurations
    ## Load preprocessing configuration
    preprocessing_config = PreprocessorConfig.from_json(preprocessing_config_path)

    ## Load training configuration
    training_config = PretrainedConfig.from_json_file(training_config_path)

    ## Load model configuration
    model_config = ModelConfig(path_to_config=model_config_path)

    # Load dataset
    dataset = load_dataset(preprocessing_config)

    # TODO: Pass to loss function in trainer
    label_weights = get_label_weights(dataset)
    label_weights_list = label_weights.tolist()
    setattr(model_config, "label_weights", label_weights_list)

    # Load model
    model = ModelsFactory.create_model(
        model_type=training_config.training_type,
        model_config=model_config,
        label_weights=label_weights
    )
    model.train()

    # Define data collator
    data_collator = CollatorFactory.create_collator(training_config.training_type)

    # Define Training Arguments
    training_args = TrainingArguments(
        # Naming
        run_name=training_config.output_model_name,
        output_dir=training_config.save_path,

        # Logging
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=training_config.logging_steps,
        report_to="wandb",

        # Batches
        per_device_train_batch_size=training_config.train_batch_size,
        per_device_eval_batch_size=training_config.evalbatch_size,
        eval_accumulation_steps=training_config.accummulation_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.epochs,

        # Optimizer
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        adam_epsilon=training_config.adam_epsilon,
        lr_scheduler_type=training_config.lr_scheduler_type,
        lr_scheduler_kwargs=training_config.lr_scheduler_kwargs,

        # Saving
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        fp16=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model evaluation metrics
    metrics = trainer.evaluate()

    csv_file = 'outputs/metrics.csv'
    model_name = training_config.output_model_name
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['model_name'] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({'model_name': model_name, **metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--model_config_path", type=str, help="Path to the model configuration file.")
    parser.add_argument("--training_config_path", type=str, help="Path to the training configuration file.")
    parser.add_argument("--preprocessing_config_path", type=str, help="Path to the preprocessing configuration file.")

    args = parser.parse_args()
    main(
        model_config_path=args.model_config_path,
        training_config_path=args.training_config_path,
        preprocessing_config_path=args.preprocessing_config_path
    )