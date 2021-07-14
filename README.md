# Classification-of-Liquid-Crystal-Phases-with-CNNs

For my final year MPhys project, I worked alongside a partner and supervisor to create deep learning models which could accept liquid crystal texture images
as an input and then correctly classify each image into either an individual phase or group of phases. We wanted to train and test our models on exclusively experimental data in order to provide a realistic and practical assessment of their ability to generalise. The dataset we needed did not exist and we thus had to obtain videos of liquid crystal phase transitions from our supervisor's PhD student, extract frames as images, transform the images, create appropriate naming conventions and avoid sharing of images across the train, validation and test sets from a given video in order to avoid data leakage. Using our overall dataset, we created three binary and two multinomial subsets which we used for classification tasks. ResNet50, Inception and a series of sequential CNN architectures were implented and then tuned on the datasets. Due to the observation of variance in the validation and test set accuracies, we assessed specific configurations of models in terms of the mean and standard deviation of the accuracies. Furthermore, we created a script which could create and then display mean and standard deviation confusion matrices. My partner, supervisor and I are currently in the process of writing a paper for publication in a journal.

Below I have provided the abstract from my final project report. For a more complete description of our work, refer to the project reports. As a final note, the 'data' folder, which contains all of the datasets we prepared, had to be left out for this remote repo due to its size. Also, a folder named 'checkpoints', which contained all of my saved models, inside the 'models and scripts' folder had to be left out for the same reason.

**Abstract:**

Convolutional neural networks and deep learning methods were used to classify images of liquid
crystal textures as specific phases. The dataset we used was built only from images produced by
polarised microscopy, which was in contrast to the use of simulated textures in previous literature. The
images included the textures of the cholesteric phase and four unique smectic phases. Five subsets of
this dataset and three network architectures, which varied in their capacity and types of layers, were
created to explore five tasks. The first three were binary classification tasks and the final two were
multinomial. The mean test accuracy values varied from 85-99%, and the standard deviation of those
values ranged from ±1-6%. Furthermore, confusion matrices were used to identify specific weaknesses
or strengths in the classification of a given phase, or class, for each task.

The results we obtained provided evidence that convolutional neural network variants could be
a viable approach for the classification of liquid crystal phases. However, we found that the highest
test accuracy values for the multinomial tasks, which were both below 90%, had potentially saturated.
Additionally, the networks trained for the binary task with the smallest dataset were observed to produce
lower test accuracy values and a higher variance in comparison to those trained for the other binary
tasks. Consequently, we believe that future work should focus on expanding the size and class balance of
the overall dataset to improve the performance of networks. This could be achieved by the creation of a
large-scale and open-source database of labelled liquid crystal texture images, which could be contributed
to by members of the liquid crystal research community


