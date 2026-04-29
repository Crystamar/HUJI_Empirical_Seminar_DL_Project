# HUJI_Empirical_Seminar_DL_Project  
**Introduction**  

Attention and Transformer based architectures such as BERT have been a staple
in the field of NLP for a while now. Recently such architectures were found to
be effective in a number of other fields of deep learning. As described in "An
image is worth 16x16 words" from 2020, BERT-like architectures have proven to
be effective in the field of image processing and classification when operating on
embedded image patches. These architectures were found to be better than big
ResNets which were the SOTA architectures for image classification at the time
when trained on large enough datasets. The hypothesized reason was that Transformers lack image-specific inductive biases that are present in convolution based
networks such as locality, two-dimensional neighborhood structure and translation
equivariance. While this makes Transformer based architectures "slower learners"
than CNNs, when trained on a sufficient amount of data they are able to reach
better results since their lack of bias allows them to learn better, more general
functions that might contradict these biases.
My hypothesis in this experiment is that changing the division of images into
patches in certain ways (using bigger patches near the middle of the image for
example), might inject image-specific bias into the model and therefore achieve
better results in the earlier phases of training. As shown in this experiment, this
might be somewhat true as when trained over CIFAR-10 most of the biased models
achieve better accuracies then the unbiased model in the first few training epochs,
but the unbiased models end up with the best accuracies after 25 training epochs.
I will try to explain this phenomenon in this paper.
This might have some significance as it can serve as a very simple and relatively
cheap way to inject bias into Transformer-based image models and achieve better
results when trained over a small dataset, or with a low number of epochs.

**Patch sizes (8, 16, 16, 8) visualized:**  
<img width="430" height="403" alt="image" src="https://github.com/user-attachments/assets/f03a7098-77b4-4199-b09d-d9a8c638d925" />


<img width="640" height="480" alt="accs_3x3" src="https://github.com/user-attachments/assets/a51a0b28-9cd8-4937-b870-78cc72c7fc0f" />


<img width="640" height="480" alt="accs_4x4" src="https://github.com/user-attachments/assets/2fbb1a91-dbdf-4d31-a224-fec2f62b4eea" />


<img width="640" height="480" alt="accs_6x6" src="https://github.com/user-attachments/assets/f806745e-dbb6-421a-a3a5-8c4d5998be46" />


<img width="640" height="480" alt="pos_similarity_patch 16_16_16" src="https://github.com/user-attachments/assets/c1c2ef04-5a4e-4437-bf38-d014cc5cad23" />


<img width="640" height="480" alt="pos_similarity_patch 19_10_19" src="https://github.com/user-attachments/assets/3cf064a5-2f27-42ec-a0fe-121a8039ff1e" />


<img width="640" height="480" alt="pos_similarity_patch 12_12_12_12" src="https://github.com/user-attachments/assets/63325489-1997-4ec6-a446-ed7e835ca26a" />


<img width="640" height="480" alt="pos_similarity_patch 8_16_16_8" src="https://github.com/user-attachments/assets/5e359f2d-8021-4909-9be8-4c49614534a6" />


<img width="640" height="480" alt="pos_similarity_patch 8_8_8_8_8_8" src="https://github.com/user-attachments/assets/07aa977e-1706-47d5-a484-5267081b4f4f" />


<img width="640" height="480" alt="pos_similarity_patch 12_8_4_4_8_12" src="https://github.com/user-attachments/assets/7dd5c1cc-7316-4e17-91d9-bac16088723e" />








