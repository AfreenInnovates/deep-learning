# Food Image Classification using Transfer Learning

## introduction

This project demonstrates the implementation of **Transfer Learning** for multi-class image classification using PyTorch and a pre-trained EfficientNet-B0 model. The goal is to classify food images into three categories: **Pizza**, **Steak**, and **Sushi**.

Transfer learning leverages knowledge gained from a model pre-trained on a large dataset (ImageNet) and adapts it to a new, smaller dataset. This approach significantly reduces training time and improves performance, especially when working with limited data.

---

## data

**Dataset Structure:**
- **Training Set:** 225 images (75 per class)
- **Test Set:** 75 images (25 per class)
- **Classes:** Pizza, Steak, Sushi
- **Organization:** Images are organized in class-based subdirectories
- Zip file: 03_computer-vision\learning\03_food-vision-transfer-learning\pizza_steak_sushi.zip
```
pizza_steak_sushi_images/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
└── test/
    ├── pizza/
    ├── steak/
    └── sushi/
```

The dataset is automatically extracted from a ZIP file and organized into this structure for easy loading with PyTorch's `ImageFolder` dataset class.

---

## methodology

### 1. Data Preparation

**Why this approach?**
The code uses PyTorch's `ImageFolder` dataset class, which automatically:
- Loads images from directory structure
- Assigns labels based on folder names
- Enables efficient batch processing

**Key Components:**
- **Path Setup:** Defines paths for data directory, ZIP file, and extracted images
- **Automatic Extraction:** Checks if data exists; if not, extracts the ZIP file
- **Directory Validation:** Verifies dataset structure and counts files in each class

### 2. Image Transformations

Two transformation approaches are demonstrated:

#### 2.1 Manual Transforms
```python
transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

**Why these specific values?**
- **Resize to 224×224:** EfficientNet-B0 expects this input size
- **ToTensor:** Converts PIL images to PyTorch tensors with values in [0, 1]
- **Normalize:** Uses ImageNet mean and std to match the pre-trained model's training distribution

#### 2.2 Auto Transforms (Preferred)
Uses the official transforms from the pre-trained model weights:
```python
auto_transforms = EfficientNet_B0_Weights.DEFAULT.transforms()
```

**Advantages:**
- Automatically applies the exact preprocessing used during pre-training
- Includes proper resizing (256) and center cropping (224)
- Uses bicubic interpolation for better quality
- Ensures consistency between training and inference

### 3. DataLoader Creation

**Purpose:** Efficiently load data in batches for training and evaluation

**Key Parameters:**
- **batch_size=32:** Process 32 images simultaneously (memory vs speed tradeoff)
- **shuffle=True (train):** Randomize training data to prevent learning order biases
- **shuffle=False (test):** Maintain consistent evaluation order
- **num_workers:** Parallel data loading using multiple CPU cores
- **pin_memory=True:** Faster GPU transfer by using page-locked memory

**Why a custom function?**
The `create_dataloaders()` function encapsulates the entire data pipeline, making it reusable and returning all necessary components (dataloaders, class names, datasets) in one call.

---

## Model Architecture

### Model Modification

**Original Classifier:**
```python
Sequential(
  Dropout(p=0.2, inplace=True)
  Linear(in_features=1280, out_features=1000, bias=True)
)
```
- Outputs 1000 classes (ImageNet categories)

**Modified Classifier:**
```python
Sequential(
  Dropout(p=0.2, inplace=True)
  Linear(in_features=1280, out_features=3, bias=True)
)
```
- Outputs 3 classes (pizza, steak, sushi)

### Transfer Learning Strategy: Feature Extraction

**Freezing the Feature Extractor:**
```python
for param in model.features.parameters():
    param.requires_grad = False
```

**Why freeze the features layer?**
- **Preserve learned patterns:** The convolutional layers already know how to detect edges, textures, shapes, and complex patterns from ImageNet
- **Reduce computation:** Only train the classifier head (thousands vs millions of parameters)
- **Prevent overfitting:** With only 225 training images, training all layers could lead to overfitting
- **Faster training:** Fewer parameters to update means faster epochs

**What gets trained?**
Only the final classifier layer adapts to our specific task of distinguishing pizza, steak, and sushi.

---

## Training Process

### Training Configuration

**Loss Function: CrossEntropyLoss**
- Appropriate for multi-class classification
- Combines LogSoftmax and NLLLoss

**Optimizer: Adam (lr=0.001)**
- Adaptive learning rate per parameter
- Combines benefits of RMSprop and momentum

**Epochs: 5**
- Sufficient for fine-tuning the classifier

### Training Loop Mechanics

**For each epoch:**
1. **Training Phase:**
   - Set model to train mode (`model.train()`)
   - Iterate through training batches
   - Forward pass: compute predictions
   - Calculate loss
   - Backward pass: compute gradients
   - Update weights using optimizer
   - Track loss and accuracy

2. **Evaluation Phase:**
   - Set model to eval mode (`model.eval()`)
   - Disable gradient computation (`torch.no_grad()`)
   - Iterate through test batches
   - Compute predictions and metrics
   - No weight updates

**Why separate train/eval modes?**
- Dropout behaves differently (active during training, inactive during evaluation)
- Batch normalization uses different statistics
- Prevents gradient computation during evaluation (saves memory)

---

## Results Analysis

### Training Metrics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.5447    | 82.22%    | 0.5412   | 86.67%  |
| 2     | 0.5110    | 88.00%    | 0.5331   | 89.33%  |
| 3     | 0.4408    | 92.00%    | 0.4906   | 90.67%  |
| 4     | 0.4074    | 94.22%    | 0.4852   | 86.67%  |
| 5     | 0.3883    | 92.89%    | 0.4464   | 88.00%  |

### Key Observations

**1. Training Progression:**
- Training loss consistently decreases (0.54 → 0.38)
- Training accuracy improves (82% → 93%)
- Shows the classifier is successfully learning to distinguish the three food types

**2. Validation Performance:**
- Best validation accuracy: **90.67%** (Epoch 3)
- Final validation accuracy: **88.00%**
- Demonstrates good generalization to unseen data

**3. Slight Overfitting Observed:**
- Training accuracy (92.89%) > Validation accuracy (88.00%)
- The gap increases in later epochs
- Epoch 4 shows validation accuracy dropping to 86.67% while training continues to improve

**Why overfitting occurs:**
- Small training dataset (225 images)
- Model complexity relative to data size
- Could be mitigated with: more data, stronger data augmentation, or early stopping

**5. Loss vs Accuracy Curves:**
- Both training and validation losses decrease steadily
- Validation loss plateaus after epoch 3
- Suggests the model has reached optimal performance for this dataset size

---

## Prediction Results

### Visualization Methods

**1. Grid Visualization:**
- Shows 16 random predictions in a 4×4 grid
- Green titles: Correct predictions
- Red titles: Incorrect predictions
- Format: "P: [Predicted] / T: [True]"

**2. Side-by-Side Comparison:**
- Shows 5 predictions with actual vs predicted labels
- Left column: Actual label (black)
- Right column: Predicted label (green/red)
- Easier to spot misclassifications

### Custom Image Predictions

**How it works:**
1. Load image from URL or local path
2. Apply the same transforms used during training
3. Pass through the model
4. Apply softmax to get probabilities
5. Select class with highest probability
6. Display image with prediction and confidence

**Denormalization:**
The visualization code reverses the normalization to display images correctly:
```python
X_sample = X_sample * std + mean
```
This converts the normalized tensors back to displayable RGB values.

### Performance on External Images

**Steak Predictions:**
- identified in most close-up shots (52-70% confidence)
- misclassified as sushi when steak is in context with people (67% as sushi)
- **Insight:** Model struggles with contextual images; works best on isolated food items

**Sushi Predictions:**
- high accuracy and confidence (74-90%)
- works well even with hands/people in frame
- **Insight:** Sushi has distinctive visual features (rolls, rice, colors) that are easier to identify

**Pizza Predictions:**
- mixed results (45-61% confidence, some misclassified as sushi)
- **Insight:** Pizza in real-world contexts is harder to classify; possible confusion with circular shapes

### why Some Predictions Fail

**1. Domain Shift:**
- Training data: Clean, centered food images
- Test images: Real-world scenarios with people, different angles, backgrounds
- The model hasn't seen this distribution during training

**2. Context Confusion:**
- Images with people eating often misclassified
- Background elements distract from the food
- The model focuses on the entire image, not just the food

**3. Limited Training Data:**
- Only 75 images per class during training
- Insufficient diversity to handle all real-world variations
- More training data with various contexts would improve robustness

**4. Visual Similarity:**
- Some pizza images might have circular shapes similar to sushi rolls
- Steak in dim lighting could resemble sushi colors
- Confidence scores (45-70%) indicate model uncertainty

---

## Key Concepts Explained

### What is Transfer Learning?

**Definition:** Using a model trained on one task (ImageNet classification) as a starting point for a different but related task (food classification).

**Benefits:**
- **Less data needed:** Can achieve good results with hundreds instead of millions of images
- **Faster training:** Only fine-tune the final layers
- **Better performance:** Leverage patterns learned from massive datasets
- **Lower cost:** Reduces computational requirements

### Feature Extraction vs Fine-Tuning

**Feature Extraction (Used here):**
- Freeze all pre-trained layers
- Only train the new classifier head
- Fast, prevents overfitting with small datasets

**Fine-Tuning (Alternative approach):**
- Unfreeze some/all pre-trained layers
- Train with very small learning rate
- Can improve performance with sufficient data
- Risk of overfitting with small datasets

### ImageNet Normalization

**Why normalize?**
- Neural networks train better with standardized inputs
- Prevents certain features from dominating due to scale
- Helps gradients flow properly during backpropagation

**Why these specific values?**
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
These are the mean and standard deviation of ImageNet dataset. Using the same normalization as the pre-trained model ensures feature compatibility.

### Batch Processing

**Why use batches?**
- **Memory efficiency:** Can't fit entire dataset in GPU memory
- **Gradient stability:** Average gradients across multiple samples
- **Parallelization:** GPUs process multiple images efficiently
- **Regularization effect:** Noise from batch selection helps generalization

**Batch size trade-offs:**
- **Larger batches (64, 128):** More stable gradients, faster training, more memory
- **Smaller batches (16, 32):** Less memory, more updates per epoch, more noise

### Dropout

**What it does:**
```python
Dropout(p=0.2)
```
Randomly sets 20% of activations to zero during training.

**Why use it?**
- Prevents co-adaptation of neurons (neurons can't rely on specific other neurons)
- Acts as ensemble learning (training multiple sub-networks)
- Reduces overfitting
- Only active during training, disabled during evaluation

---

## Conclusions

### Strengths

1. **Effective Transfer Learning:**
   - Achieved 88% validation accuracy with only 225 training images
   - Training took only 5 epochs (~1-2 minutes on GPU)
   - Demonstrates the power of pre-trained models

3. **Strong Performance on Isolated Food:**
   - High confidence and accuracy on close-up, centered food images
   - Sushi classification particularly robust (74-90% confidence)

### Weaknesses

1. **Limited Generalization:**
   - Struggles with real-world images containing people and contexts
   - Lower confidence on images with complex backgrounds
   - Domain gap between training and test distributions

2. **Small Dataset:**
   - Signs of overfitting in later epochs
   - Insufficient diversity to handle all variations

### Lessons Learned

**1. Transfer Learning Requirements:**
- Must use same preprocessing as pre-trained model
- Feature extraction works well with limited data
- Careful selection of what layers to freeze/unfreeze

**2. Model Evaluation:**
- Test on diverse, real-world images, not just validation set
- Confidence scores provide insight into model uncertainty
- Visual inspection reveals failure modes metrics don't show

**3. Overfitting Indicators:**
- Growing gap between train and validation accuracy
- Validation loss plateauing or increasing
- Early stopping could improve final performance

---

## Potential Improvements

### 1. Data Augmentation
```python
transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224),
])
```
**Why:** Artificially expands dataset, improves robustness to variations

### 2. Larger Dataset
- Collect more images (500-1000 per class)
- Include diverse contexts (people eating, different angles, lighting)
- Balance class distributions

### 3. Advanced Transfer Learning
- Unfreeze last few layers of feature extractor
- Use differential learning rates (lower for pre-trained layers)
- Fine-tune on food-specific pre-trained model (e.g., Food-101)

### 4. Model Ensemble
- Train multiple models with different architectures
- Average predictions for better robustness
- Reduces impact of individual model biases

### 5. Class Activation Mapping (CAM)
- Visualize which parts of images the model focuses on
- Debug misclassifications
- Ensure model learns correct features

### 6. Hyperparameter Tuning
- Learning rate: Try 0.0001, 0.001, 0.01
- Batch size: Experiment with 16, 32, 64
- Dropout rate: Test 0.3, 0.4, 0.5
- Use learning rate scheduling

### 7. Early Stopping
```python
if val_loss hasn't improved for 3 epochs:
    stop training and use best model
```
**Why:** Prevents overfitting, saves computation time

---

## Technical Requirements

### Environment
- **GPU:** NVIDIA T4 (or similar) with CUDA support
- **Framework:** PyTorch with torchvision
- **Platform:** Google Colab or local GPU environment

### Dependencies
```python
torch               
torchvision        
torchinfo           
tqdm
matplotlib          
PIL                 
requests             
```
---

## License & Usage

This notebook is an educational resource for learning

Feel free to adapt the code for your own classification tasks by:
1. Replacing the dataset with your target classes
2. Adjusting the final layer to match your number of classes
3. Following the same training and evaluation workflow

---

**Author Note:** This README provides comprehensive explanations of code concepts, design decisions, and results interpretation.