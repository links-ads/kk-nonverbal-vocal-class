# Multimodal Model

This repository contains the implementation of a multimodal model that combines audio and text data for enhanced performance.

## Architecture
```
+-------------+     +-------------+
|  Audio      |     |  Text       |
|  Model      |     |  Model      |
+-------------+     +-------------+
    |                 |
  +---+             +---+
  |Neck|             |Neck|
  +---+             +---+
    |                 |
    +--------+--------+
             |
          +--------+
          | Fusion |
          | Module |
          +--------+
             |
        +----------+
        |Classifier|
        +----------+
```

## Components

1. **Audio Model**: Processes audio input and extracts relevant features.
2. **Text Model**: Processes text input and extracts relevant features.
3. **Neck**: A bridging module that connects the audio and text models to the fusion module.
4. **Fusion Module**: Combines the features from both the audio and text models to produce the final output.
5. **Classifier**: Combines the fused features to make the final prediction.
