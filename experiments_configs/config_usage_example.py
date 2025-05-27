"""
Example usage of the Config class

This example demonstrates how to use the Config class to load and manage configurations.
"""
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from non_verbal_voc_class.configs import Config


def main():
    print("Config Class Usage Examples")
    print("=" * 50)
    
    # Example 1: Load configuration from JSON file
    print("\n1. Loading configuration from JSON file:")
    config_path = os.path.join(os.path.dirname(__file__), 'config_example.json')
    
    try:
        config = Config.from_json(config_path)
        print(f"✓ Configuration loaded successfully")
        print(f"✓ Model config present: {config.model_config is not None}")
        print(f"✓ Preprocessing config present: {config.preprocessing_config is not None}")
        print(f"✓ Training config present: {config.training_config is not None}")
        
        # Access specific configuration values
        if config.model_config:
            print(f"  - Model type: {config.model_config.model_type}")
            print(f"  - Audio model: {config.model_config.audio_model_name}")
            print(f"  - Finetune method: {config.model_config.finetune_method}")
        
        if config.preprocessing_config:
            print(f"  - Target sampling rate: {config.preprocessing_config.target_sampling_rate}")
            print(f"  - Max duration: {config.preprocessing_config.max_duration}")
        
        if config.training_config:
            print(f"  - Batch size: {config.training_config.batch_size}")
            print(f"  - Learning rate: {config.training_config.learning_rate}")
            print(f"  - Number of epochs: {config.training_config.num_epochs}")
            
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
    
    # Example 2: Create configuration from dictionary
    print("\n2. Creating configuration from dictionary:")
    config_dict = {
        "model_config": {
            "model_type": "wav2vec2",
            "audio_model_name": "facebook/wav2vec2-base-960h",
            "finetune_method": "lora",
            "lora_rank": 16
        },
        "training_config": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 10
        }
    }
    
    config_from_dict = Config.from_dict(config_dict)
    print(f"✓ Configuration created from dictionary")
    print(f"✓ Model type: {config_from_dict.model_config.model_type}")
    print(f"✓ Batch size: {config_from_dict.training_config.batch_size}")
    
    # Example 3: Update configuration values
    print("\n3. Updating configuration values:")
    config_from_dict.update_model_config(num_labels=5, use_weighted_layer_sum=False)
    config_from_dict.update_training_config(batch_size=64, learning_rate=1e-3)
    
    print(f"✓ Updated num_labels: {config_from_dict.model_config.num_labels}")
    print(f"✓ Updated batch_size: {config_from_dict.training_config.batch_size}")
    
    # Example 4: Save configuration to JSON
    print("\n4. Saving configuration to JSON:")
    output_path = os.path.join(os.path.dirname(__file__), 'output_config.json')
    config_from_dict.to_json(output_path)
    print(f"✓ Configuration saved to: {output_path}")
    
    # Example 5: Validate configuration
    print("\n5. Validating configuration:")
    is_valid = config.validate()
    print(f"✓ Configuration is valid: {is_valid}")
    
    # Example 6: Create empty config and add components
    print("\n6. Creating empty config and adding components:")
    empty_config = Config()
    empty_config.update_model_config(
        model_type="whisper",
        audio_model_name="openai/whisper-base"
    )
    empty_config.update_preprocessing_config(
        target_sampling_rate=16000,
        max_duration=30.0
    )
    
    print(f"✓ Empty config created and populated")
    print(f"✓ Model type: {empty_config.model_config.model_type}")
    print(f"✓ Sampling rate: {empty_config.preprocessing_config.target_sampling_rate}")
    
    print(f"\n{config}")


if __name__ == "__main__":
    main()
