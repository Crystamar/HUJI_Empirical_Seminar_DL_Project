# HUJI_Empirical_Seminar_DL_Project  
**Introduction**  

Attention and Transformer based architectures such as BERT have been a staple
in the eld of NLP for a while now. Recently such architectures were found to
be eective in a number of other elds of deep learning. As described in "An
image is worth 16x16 words" from 2020, BERT-like architectures have proven to
be eective in the eld of image processing and classication when operating on
embedded image patches. These architectures were found to be better than big
ResNets which were the SOTA architectures for image classication at the time
when trained on large enough datasets. The hypothesized reason was that Transformers lack image-specic inductive biases that are present in convolution based
networks such as locality, two-dimensional neighborhood structure and translation
equivariance. While this makes Transformer based architectures "slower learners"
than CNNs, when trained on a sucient amount of data they are able to reach
better results since their lack of bias allows them to learn better, more general
functions that might contradict these biases.
My hypothesis in this experiment is that changing the division of images into
patches in certain ways (using bigger patches near the middle of the image for
example), might inject image-specic bias into the model and therefore achieve
better results in the earlier phases of training. As shown in this experiment, this
might be somewhat true as when trained over CIFAR-10 most of the biased models
achieve better accuracies then the unbiased model in the rst few training epochs,
but the unbiased models end up with the best accuracies after 25 training epochs.
I will try to explain this phenomenon in this paper.
This might have some signicance as it can serve as a very simple and relatively
cheap way to inject bias into Transformer-based image models and achieve better
results when trained over a small dataset, or with a low number of epochs.

**Patch sizes (8, 16, 16, 8) visualized:**  
<img width="430" height="403" alt="image" src="https://github.com/user-attachments/assets/f03a7098-77b4-4199-b09d-d9a8c638d925" />

