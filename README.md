Since Dataset was imbalanced, confusion matrix wasn't good
but still model was shown to be least confused
but model was really underconfident, after all what do you expect of a difference of 100 and 2000 images in two different categories

Model Used : GoogleNet Inception (miniaturized by me)

## RESULTS : 

After 20 EPOCHS : <br>

209/209 - 4s - 19ms/step - **accuracy**: 0.7048 - **f1_score**: 0.7352 - loss: 1.1292 - **precision**: 0.9348 - **recall**: 0.4176 - **top_5_acc**: 0.9441

<img src="confusion_matrix.png" alt="confusion_matrix" width="500">

## Tensorboard Profile

### Accuracy over Epochs 
<img src="tensorboard/accuracy.png" alt="confusion_matrix" width="500">

### Precision over Epochs
<img src="tensorboard/precision.png" alt="confusion_matrix" width="500">

### Recall over Epochs
<img src="tensorboard/recall.png" alt="confusion_matrix" width="500">

### F1-score over Epochs
<img src="tensorboard/f1.png" alt="confusion_matrix" width="500">

### TOP-5 Accuracy over Epochs
<img src="tensorboard/top5.png" alt="confusion_matrix" width="500">
