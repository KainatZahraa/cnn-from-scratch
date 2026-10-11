# CIFAR-10 Classification from Scratch

Data Learning for Perception — Question 1

This project investigates how augmentation and regularization affect a small CNN trained from scratch on limited CIFAR-10 data. Models are evaluated on clean images and a fixed corrupted test set.

## Files

- `notebook.ipynb`: Executed notebook containing Parts A–F, results, plots, and analysis.
- `README.md`: Project overview and reproduction instructions.

## Dataset and Experimental Setup

CIFAR-10 is loaded using `tensorflow.keras.datasets.cifar10.load_data()`.

| Split | Total images | Images per class |
|---|---:|---:|
| Selected training subset | 500 | 50 |
| Training | 400 | 40 |
| Validation | 100 | 10 |
| Clean test | 10,000 | 1,000 |

- Sampling and splitting seed: **42**
- Training seeds: **10, 42, 74**
- Corruption seed: **2026**
- Pixel values normalized to `[0, 1]`
- No pretrained weights or additional training images
- Hyperparameters and final configuration selected using validation data only
- Test evaluation performed after model choices were finalized

Results are reported as mean ± sample standard deviation across the three training seeds.

## Model and Training

The network includes parallel Inception-style branches:

- 1×1 convolution
- 1×1 reduction followed by a 3×3 convolution
- 1×1 reduction followed by a 5×5 convolution

Branch outputs are concatenated and followed by further convolution, pooling, Flatten, and Dense layers.

| Model | Trainable parameters |
|---|---:|
| Plain baseline | 142,090 |
| Regularized model | 142,378 |

The regularized model uses:

- BatchNorm after every convolution and the hidden Dense layer, before ReLU; momentum **0.9**
- Dropout **0.3** after the hidden layer's ReLU
- AdamW with weight decay **0.0001**
- Batch size **32**
- Maximum **50 epochs**
- Early stopping on validation loss, patience **15**, restoring the best weights
- Learning rates of **0.001**, **0.0003**, and **0.0001** for epochs 1–25, 26–40, and 41–50

The Part A baseline uses Adam and 50 epochs without augmentation or the added regularization.

## Augmentation

All training augmentations are generated on the fly.

- **Geometric:** padding by four pixels, random 32×32 crop, horizontal flip, rotation within ±25°
- **Photometric:** brightness ±0.15, contrast and saturation factors 0.8–1.2, occasional 3×3 average blur and Gaussian noise
- **Cutout:** a 10×10 black square with 50% probability
- **Mixup:** alpha 0.2, mixing both images and labels
- **Label smoothing:** 0.05 in the Mixup configuration only

Validation images receive no random augmentation.

## Shifted-Test Pipeline

The fixed pipeline transforms the 10,000 clean test images using:

- Rotation within ±25°
- Contrast factors between 0.7 and 1.3
- Brightness changes between −0.2 and +0.2
- Random rectangular occlusion covering approximately 25% of the image
- Grayscale conversion with 50% probability

Labels and image order are preserved. Blur and noise are used in training augmentation but are not included in this implemented test pipeline.

Grayscale is held out from explicit training augmentation. Because it was implemented by the experimenter, it is not a genuinely blind corruption. A separate grayscale-only evaluation converts all test images to grayscale.

## Ablation Results

All Part D rows use the regularized model. Consequently, its “No augmentation” row is different from Part A's plain baseline.

| Configuration | Validation | Clean test | Shifted test |
|---|---:|---:|---:|
| No augmentation | 43.67% ± 4.93% | 36.94% ± 1.44% | 17.76% ± 0.40% |
| Geometric only | 40.67% ± 4.16% | 39.83% ± 3.60% | 20.42% ± 0.92% |
| Photometric only | 39.67% ± 2.08% | 38.37% ± 1.22% | 18.45% ± 0.71% |
| Geometric + photometric | **45.00% ± 2.00%** | **42.72% ± 0.12%** | 22.61% ± 0.92% |
| Above + Cutout | 43.00% ± 3.61% | 41.15% ± 1.30% | **27.26% ± 0.46%** |
| Above + Mixup + label smoothing | 43.00% ± 3.46% | 40.62% ± 1.20% | 27.20% ± 0.52% |

**Geometric + photometric** was selected using mean validation accuracy. Its three seed models were retained separately, not combined into an ensemble.

Cutout produced the largest incremental shifted-test gain: **4.65 percentage points** over geometric + photometric augmentation. The final selection was not changed after viewing test results.

The plain Part A baseline achieved:

- Clean test: **36.24% ± 0.93%**
- Shifted test: **14.79% ± 0.92%**

## Error Analysis and Test-Time Augmentation

Part E includes per-class recall, a mean row-normalized confusion matrix, and eight misclassified examples.

The final configuration achieved **32.55% ± 1.71%** on the grayscale-only test, a drop of **10.17 percentage points** from clean-test accuracy.

Part F averages softmax probabilities over four views:

1. Original image
2. Horizontal flip
3. One-pixel shift left
4. One-pixel shift right

| Inference method | Shifted-test accuracy | Time for 10,000 images |
|---|---:|---:|
| Ordinary | 22.61% ± 0.92% | 0.69 ± 0.03 seconds |
| Four-view TTA | 23.28% ± 1.14% | 2.85 ± 0.03 seconds |

TTA improved accuracy by **0.67 ± 0.23 percentage points**, at approximately **4.17×** the measured runtime. Timing includes view generation and probability averaging and was measured once per seed.

## How to Reproduce

### Recommended: Kaggle

1. Download `notebook.ipynb` and import it into a Kaggle notebook.
2. Enable a GPU accelerator. The recorded runs used an NVIDIA Tesla T4.
3. Enable Internet access for the initial CIFAR-10 download.
4. Run the notebook using the execution order below.
5. Keep all model choices fixed when evaluating the test sets.

Required libraries:

- TensorFlow/Keras
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- pandas

### Local Setup

Install the required packages in a compatible Python environment:

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib pandas jupyter
```

Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

GPU acceleration is recommended. CPU execution may take longer.

### Execution Order

1. Run imports and Part A's data preparation, baseline training, summaries, and learning curves.
2. Construct the fixed clean and shifted test arrays.
3. **Skip Part A's cell beginning `#clean and shifted test accuracy analysis` for now.**
4. Run Parts B and C to define augmentations, the regularized model, and training settings.
5. Run Part D's training, validation comparison, and final configuration selection.
6. Once choices are finalized, run Part D's test evaluation and table.
7. Return to the deferred Part A test-evaluation cell.
8. Run Parts E and F for error analysis and TTA.

The notebook retains trained models in memory. Restarting the kernel loses them unless they have been saved separately.

### Reproducibility Notes

The sampling, corruption, and training seeds are fixed. Exact package versions were not recorded in the submitted experiment, and deterministic TensorFlow operations were not explicitly enabled. Small numerical differences may therefore occur across reruns, hardware, and software versions.

Saved notebook outputs document the reported runs. Reproduction should follow the frozen configuration without using test results to retune the model.

## Main Conclusion

Augmentation and regularization improved robustness compared with the plain baseline, but substantial errors remained under combined corruptions. Validation-based model selection favored geometric + photometric augmentation, while Cutout showed the largest incremental gain on the shifted-test ablation. TTA provided a further modest improvement at increased inference cost.
