### Applying Machine Learning Algorithms to Detect Continuous Gravitational Waves
This is a showcase of a directed research data science project (DATA 390) that has a focus of advancing the student’s understanding of data science by applying machine learning concepts to an area of interest - astrophysics.

##### Astronomical Background
Every mass has an effect on every other mass in the universe, and understanding the constant change in this effect, or changes in gravity, can give us deeper insight on the behavior of celestial bodies and other concepts in the realm of outer space. Fluctuations in gravity can be called gravitational waves and are essentially ripples, or the rapid stretching and compression, of spacetime. Being able to detect gravitational waves will allow us to improve physics theories and might even provide answers to unknowns such as the source of dark matter.
In 2015, the first detection of a gravitational wave occurred from the source of two colliding black holes. This signal was a compact binary coalescence (CBC) signal, the first and only category of gravitational wave detected so far. At present, gravitational wave signals can be classified into four categories: CBC, burst, continuous, and stochastic. We are interested in continuous signals, which are weak and long-lasting.
Continuous gravitational waves have yet to be detected despite the technological strides made in the astrophysics field, but recent efforts utilizing certain machine learning algorithms are starting to make the hope of detection a reality.

##### Objective
The goal of this Kaggle competition is to detect continuous gravitational-wave signals by developing a model sensitive enough to detect weak yet long-lasting signals emitted by rapidly-spinning neutron stars within noisy data.

##### Data
The data used for this project was found free from an online data analysis competition provided by the European Gravitational Observatory - the Kaggle competition "G2Net Detecting Continuous Gravitational Waves"(https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/overview). The data consists of time-frequency data from the two Laser Interferometer Gravitational-Wave Observatory (LIGO) interferometers, some of the samples containing injected, simulated continuous gravitational wave signals.

The data is in an open source fromat, Hierarchical Data Format version 5 (HDF5), which is commonly used to support large, complex, heterogeneous data. The HDF5 strucutre includes two major object types:
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

In this approach, it was decided that developing spectrograms from this data would lead to the most intersting findings. In these spectrograms, the timestamps are represented on the y-axis, the frequencies are represented on the x-axis, and the SFT amplitudes for each frequency at every timestep are represented in a colormesh. These spectrograms have 'real parts' and 'imaginary parts', but in this process, 'imaginary parts' are of no concern.

![Alt text](url_to_spectrogram_image)

As seen below, the spectrograms are incredibly noisy, and the human eye cannot detect the presence of one of these simulated waves.

![Alt text](url_to_presence_image)

##### Process
