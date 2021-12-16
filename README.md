# fine-grained-car-classification
fine-grained-car-classification

# Attempts and Results
| Method                  | Accuracy          | Note                                                                                               |
| -------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------- |
| ResNet101 Baseline         | 0.9272478547444347 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=32 |
| ResNet50+ArcFace+FocalLoss | 0.9322223604029349 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=64 |
| ResNet50+MoCo(RandomColorJitter+RandomGrayscale+RandomHorizontalFlip) | 0.8977739087178211 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=32 |
