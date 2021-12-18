# fine-grained-car-classification
fine-grained-car-classification

# Attempts and Results
| Method                  | Accuracy          | Note                                                                                               |
| -------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------- |
| ResNet101 Baseline         | 0.9240144260664096 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=30, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=32 |
| ResNet50+ArcFace+FocalLoss | 0.9322223604029349 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=64, s=30.0, m=0.50, easy_margin=False, gamma=2 |
| ResNet50+MoCo(RandomColorJitter+RandomGrayscale+RandomHorizontalFlip) | 0.8977739087178211 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=32 |
| ResNet50+TripletMarginLoss | 0.8865812709861958 | input_size=(448, 448), optim=SGD, lr=0.1, epochs=100, lr_decay_rate=0.1, lr_decay_step=10(by epochs), batch_size=32, margin=5.0, lambda_triplet=0.1 |
