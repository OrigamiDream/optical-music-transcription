# Optical Music Transcription

Giving machine an ability that is able to transcript raw music sheets.

## Known Issues and Improvement Plans
- All output rows are exactly same — Due to **Gradient Vanishing**
- `changed-N` and `velocity-N` features must be separated with different losses and metrics.
  - `changed-N`: loss=`binary-crossentropy`, activations=`sigmoid`
  - `velocity-N`: loss=`mean-squared-error`, activations=`linear`