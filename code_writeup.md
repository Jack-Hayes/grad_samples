## Applying Machine Learning Algorithms to Detect Continuous Gravitational Waves
This is a showcase of a directed research data science project (DATA 390) that has a focus of advancing the student’s understanding of data science by applying machine learning concepts to an area of interest - astrophysics.

#### Astronomical Background
Every mass has an effect on every other mass in the universe, and understanding the constant change in this effect, or changes in gravity, can give us deeper insight on the behavior of celestial bodies and other concepts in the realm of outer space. Fluctuations in gravity can be called gravitational waves and are essentially ripples, or the rapid stretching and compression, of spacetime. Being able to detect gravitational waves will allow us to improve physics theories and might even provide answers to unknowns such as the source of dark matter.
In 2015, the first detection of a gravitational wave occurred from the source of two colliding black holes. This signal was a compact binary coalescence (CBC) signal, the first and only category of gravitational wave detected so far. At present, gravitational wave signals can be classified into four categories: CBC, burst, continuous, and stochastic. We are interested in continuous signals, which are weak and long-lasting.
Continuous gravitational waves have yet to be detected despite the technological strides made in the astrophysics field, but recent efforts utilizing certain machine learning algorithms are starting to make the hope of detection a reality.

#### Objective
The goal of this Kaggle competition is to detect continuous gravitational-wave signals by developing a model sensitive enough to detect weak yet long-lasting signals emitted by rapidly-spinning neutron stars within noisy data.

#### Data
The data used for this project was found free from an online data analysis competition provided by the European Gravitational Observatory - the Kaggle competition "G2Net Detecting Continuous Gravitational Waves"(https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/overview). The data consists of time-frequency data from the two Laser Interferometer Gravitational-Wave Observatory (LIGO) interferometers, some of the samples containing injected, simulated continuous gravitational wave signals.
These two interferometers, Hanford (H1) and Livingston (L1), are located in Washington State and Louisiana respectively. They are able to detect the smallest of disturbances on Earth. If a gravitational wave were to 'pass through' Earth, rapidly stretching and compressing the planet at a very small scale, the detectors are able to record nearly the same exact distrurbance at essentially the same time. Being able to extract this data from L1 and H1 and identify patterns in both detectors when a simulated gravitational wave is present is the aim of this analysis.

The data is in an open source format, Hierarchical Data Format version 5 (HDF5), which is commonly used to support large, complex, heterogeneous data. The HDF5 strucutre includes two major object types:
Datasets: Typed multidimensional arrays
Groups: Container structures that can hold datasets and other groups
Below is the formatting of the HDF5 files in this specific context.
ID: top group of the HDF5 file and links the datapoint to it's label in the train_labels csv (group)
frequency_Hz: range frequencies measured by the dectors (dataset)
H1: contains the data for the LIGO Hanford decector (group)
	SFTs: Short-time Fourier Transforms amplitudes for each timestamp at each frequency (dataset)
	timestamps contains: timestamps for the measurement (dataset)
L1: contains the data for the LIGO Livingston decector (group)
	SFTs: Short-time Fourier Transforms amplitudes for each timestamp at each frequency (dataset)
	timestamps contains: timestamps for the measurement (dataset)

![Alt text](url_to_structure_image)

In this approach, it was decided that developing spectrograms from this data would lead to the most intersting findings. In these spectrograms, the timestamps are represented on the y-axis, the frequencies are represented on the x-axis, and the SFT amplitudes for each frequency at every timestep are represented in a colormesh. These spectrograms have 'real parts' and 'imaginary parts', but in this process, 'imaginary parts' are of no concern. Keep in mind L1 and H1 have their own respective spectrograms for each sample.

![Alt text](url_to_spectrogram_image)

As seen below, the spectrograms are incredibly noisy, and the human eye cannot detect the presence of one of these simulated waves.

![Alt text](url_to_presence_image)

#### Process
The development of a densely connected convolutional neural network (https://arxiv.org/abs/1608.06993), or DenseNet, using the PyTorch library in Python was decided on for this project as seen in code_sample(url). This is a noisy binary image classification problem, label=1 when a wave is present and label=0 when a wave is not present. To preprocess the data for the DenseNet, a function was written to first extract the data from its HDF5 format. A minmax scale (equation) is manually applied to the SFT data before the spectrograms are created. 
The user has control on 'splitting' the spectrograms per sample. This 'splitting' essentially slices a spectrogram sample into a user-defined amount of smaller spectrograms. In analysis, these split spectrograms are recognized as the same sample and their classification soft probabilities are averaged to decide the classification of the sample based on a user-defined cutoff value. An example of splitting a sample into four different spectrograms is shwon below

![Alt text](url_to_unsplit_image)
![Alt text](url_to_split_image)

In this process, it was decided that the user not split spectrograms. Although there is more data generated for training and predicting each sample, the 'uniqueness' of the sinusoidal waves of interest are diminshed against the noisy background even more so when the signal is 'split'.

Image augmentation is also used to denoise the data. In the training data, each sample will go through a user-defined transformation - a horizontal shift in this case. So during the training phase, for each spectrogram sample, a horizontally translated spectrogram of that same sample will be fed into the model. Below is a representation of what this specific augmentation looks like.

![Alt text](url_to_augment_image)

The DenseNet model itself follows the architecture described in the hyperlinked paper at the top of this section.

To calculate testing accuracy, the model returns accuracy scores for L1 and H1 samples separately as the model is trained on these samples separately. This will be changed.

Lastly, there were some miscellaneous considerations when preprocessing the data. The creators of this Kaggle competition created 'easter egg' data that was given to the competitors. These 'easter eggs' have a label of -1 and were not processed in this approach as they do not contain relevant data, but rather fun pictures. There were also a few outlier samples which had significantly smaller timestep counts in the hundreds or low thousands, where the rest of the data had timestamp counts greater than 4096. With a user-defined threshold of 4096, these outliers were excluded as well. 

#### Future
The main concern of this project at present is obtaining the computational resources necessary to run this process over the entire dataset. The current plan is to move the data and script to William & Mary's HPCs for the analysis. This issue also shines a light on the problem of maximizing the efficiency of this process in terms of memory usage which will also be addressed.

For the model itself, hyperparameter optimization will come after the previous issues are fixed.
