# All-fiber microendoscopic polarization sensing at single-photon level aided by deep-learning

This repository provides data and supplementary material to the paper **All-fiber microendoscopic polarization sensing at single-photon level aided by deep-learning**, by Martin Bielak, Dominik Vašinka, and Miroslav Ježek. <br>
Currently available at [arXiv:2405.02172 [physics.optics]](https://arxiv.org/abs/2405.02172)

The provided files contain Python (3.11) codes written as Jupyter notebooks to reproduce the results presented in the paper, along with the necessary data stored as Numpy (1.24.3) arrays. The deep learning models were built using the Keras (2.12.0) and Tensorflow (2.12.0) libraries.

## List of code files in this repository:
### Main_results.ipynb
A Jupyter notebook reproducing the (in)fidelity evaluation on the test set data samples. <br>
Stored in the main directory.  <br>
Requires the following files:
- Data pairs of experimentally acquired count distributions with their corresponding polarization coherence matrices stored at "Data_sets/Performance_evaluation/Performance_evaluation_50_ms_data.npz"
- A deep learning model stored at "Trained_models/Main_results_model.h5"

### Graph_Number_of_channels.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 2 (a) - dependence of infidelity on the varying numbers of active channels. <br>
Stored in the main directory. <br>
Requires the following files:
- Data pairs of experimentally acquired count distributions with their corresponding polarization coherence matrices stored at "Data_sets/Performance_evaluation/Performance_evaluation_50_ms_data.npz"
- Deep learning models, each utilizing a unique combination of active detection channels, stored in the folder "Trained_models/Number_of_Channels"
- A list of active detection channels for each of these deep learning models stored at "Used_channel_combinations.npy"

### Graph_Repetition_rate.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 2 (b) - dependence of infidelity on the measurement repetition rate and the collective number of detected photons. <br>
Stored in the main directory. <br>
Requires the following files:
- All files stored in the folder "Data_sets/Performance_evaluation" containing data pairs for varying acquisition times (i.e., the inverse of the repetition rate).
- A deep learning model stored at "Trained_models/Main_results_model.h5"

### Dense_connective_tissue.ipynb
A Jupyter notebook reproducing the false-colored polarization reconstruction of the dense connective tissue scan shown in the right panel of Fig. 3. <br>
Stored in the main directory. <br>
Requires the following files:
- Measured count distributions for each pixel of the polarization scan stored at "Data_sets/Scan_Dense_connective_tissue/Dense_connective_tissue_data.npz"
- A deep learning model stored at "Trained_models/Scan_Dense_connective_tissue_model.h5"

### Graph_Diatom.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 4 - comparison of the polarization state time evolution with and without a diatom moving in front of the all-fiber sensor. <br>
Stored in the main directory. <br>
Requires the following files:
- Measured count distributions for each pixel of the diatom sensing stored at "Data_sets/Scan_Diatom/Diatom_data.npz"
- A deep learning model stored at "Trained_models/Scan_Diatom_model.h5"

### Supporting_func_file.py
A Python file containing the definitions of functions, which are called by the files listed above. <br>
Stored in the main directory.


## List of files containing a deep learning model in this repository:
### Main_results_model.h5
The model used to evaluate the performance of the all-fiber sensor using the test set data samples, and to reproduce the graph shown in panel (b) of Fig. 2. <br>
Stored in the folder "Trained_models" <br>
Called by the codes:
- Main_results.ipynb
- Graph_Repetition_rate.ipynb

### Scan_Dense_connective_tissue_model.h5
The model used to reconstruct the polarization states in each pixel of the dense connective tissue scan. <br>
Stored in the folder "Trained_models" <br>
Called by the codes:
- Dense_connective_tissue.ipynb

### Scan_Diatom_model.h5
The model used to reconstruct the polarization states in each bin of the moving diatom scan, shown in Fig 4. <br>
Stored in the folder "Trained_models" <br>
Called by the codes:
- Graph_Diatom.ipynb

### Channels_dependency_[X]_channels.h5
The series of models used to reproduce the graph shown in panel (a) of Fig. 2. The "X" in the name is a placeholder for the actual number separating each model. <br>
Stored in the folder "Trained_models/Number_of_Channels" <br>
Called by the codes:
- Graph_Number_of_channels.ipynb


## List of data files in this repository:
### Performance_evaluation_X_ms_data.npz
A series of data files containing pairs of measured count distributions with their corresponding polarization coherence matrices. The "X" in the name is a placeholder for the actual number of milliseconds representing the acquisition time. These data files are used to reproduce the test set infidelity evaluation and the two graphs shown in Fig 2. <br>
Stored in the folder "Data_sets/Performance_evaluation" <br>
Called by the codes:
- Main_results.ipynb (50 ms file only)
- Graph_Number_of_channels.ipynb (50 ms file only)
- Graph_Repetition_rate.ipynb (all ms versions)

### Dense_connective_tissue_data.npz
A data file containing measured count distributions for each pixel of the polarization scan of dense connective tissue, shown in Fig. 3. <br>
Stored in the folder "Data_sets/Scan_Dense_connective_tissue" <br>
Called by the codes:
- Dense_connective_tissue.ipynb

### Diatom_data.npz
A data file containing measured count distributions for each time bin of the polarization scan of a moving diatom, shown in Fig. 4. <br>
Stored in the folder "Data_sets/Scan_Diatom" <br>
Called by the codes:
- Graph_Diatom.ipynb

### Used_channel_combination.npy
A Numpy array containing a list of active detection channels for each deep learning model used for reproducing the graph shown in the (a) panel of Fig. 2. <br>
Stored in the main directory. <br>
Called by the codes:
- Dense_connective_tissue.ipynb

















