# All-fiber microendoscopic polarization sensing at single-photon level aided by deep-learning

This repository provides data and supplementary material to the paper **All-fiber microendoscopic polarization sensing at single-photon level aided by deep-learning**, by Martin Bielak, Dominik Vašinka, and Miroslav Ježek.

The provided files contain Python (3.11) codes written as Jupyter notebooks to reproduce the results presented in the paper along with the necessary data stored as Numpy (1.24.3) arrays. The deep learning models were built using the Keras (2.12.0) and Tensorflow (2.12.0) libraries.

## List of code files in this repository:
### Main_results.ipynb
A Jupyter notebook reproducing the (in)fidelity evaluation on the test set data samples.
Stored at the main directory.
Requires following files:
- Data pairs of experimentally acquired count distributions with their corresponding polarization coherence matrices stored at "Data_sets/Performance_evaluation/Performance_evaluation_50_ms_data.npz"
- A deep learning model stored at "Trained_models/Main_results_model.h5"

### Graph_Number_of_channels.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 2 (a) - dependence of infidelity on the varying numbers of active channels.
Stored at the main directory.
Requires following files:
- Data pairs of experimentally acquired count distributions with their corresponding polarization coherence matrices stored at "Data_sets/Performance_evaluation/Performance_evaluation_50_ms_data.npz"
- Deep learning models, each utilizing a unique combination of active detection channels, stored at the folder "Trained_models/Number_of_Channels"
- A list of active detection channels for each these deep learning models stored at "Used_channel_combinations.npy"

### Graph_Repetition_rate.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 2 (b) - dependence of infidelity on the measurement repetition rate and collective number of detected photons.
Stored at the main directory.
Requires following files:
- All files stored at the folder "Data_sets/Performance_evaluation" containing data pairs for varying acquisition times (i.e., inverse of the repetition rate).
- A deep learning model stored at "Trained_models/Main_results_model.h5"

### Dense_connective_tissue.ipynb
A Jupyter notebook reproducing the false-colored polarization recontruction of the dense connective tissue scan shown in the right panel of Fig. 3.
Stored at the main directory.
Requires following files:
- Measured count distributions for each pixel of the polarization scan stored at "Data_sets/Scan_Dense_connective_tissue/Dense_connective_tissue_data.npz"
- A deep learning model stored at "Trained_models/Scan_Dense_connective_tissue_model.h5"

### Graph_Diatom.ipynb
A Jupyter notebook reproducing the graph shown in Fig. 4 - comparison of the polarization state time evolution with and without a diatom moving in front of the all-fiber sensor.
Stored at the main directory.
Requires following files:
- Measured count distributions for each pixel of the polarization scan stored at "Data_sets/Scan_Dense_connective_tissue/Dense_connective_tissue_data.npz"
- A deep learning model stored at "Trained_models/Scan_Dense_connective_tissue_model.h5"

### Supporting_func_file.py
A Python file containing the definitions of functions called by the files listed above.
Stored at the main directory.


## List of files containing a deep learning model in this repository:
### Main_results_model.h5
The model used to evaluate the performance of the all-fiber sensor using the test set data samples, and to reproduce the graph shown in panel (b) of Fig. 2.
Stored at the folder "Trained_models"
Called by the codes:
- Main_results.ipynb
- Graph_Repetition_rate.ipynb

### Scan_Dense_connective_tissue_model.h5
The model used to reconstruct the polarization states in each pixel of dense connective tissue scan.
Stored at the folder "Trained_models"
Called by the codes:
- Dense_connective_tissue.ipynb

### Scan_Diatom_model.h5
The model used to reconstruct the polarization states in each time bin of the moving diatom scan, shown in Fig 4.
Stored at the folder "Trained_models"
Called by the codes:
- Dense_connective_tissue.ipynb

### Channels_dependency_[X]_channels.h5
The series of models used to reproduce the graph shown in panel (a) of Fig. 2. The "X" in the name is a placeholder for the actual number seperating each model.
Stored at the folder "Trained_models/Number_of_Channels"
Called by the codes:
- Graph_Number_of_channels.ipynb


## List of data files in this repository:
### Performance_evaluation_X_ms_data.npz
A series of data files containing pairs of measured count distributions with their corresponding polarization coherence matrices. The "X" in the name is a placeholder for the actual number of miliseconds representing the acquisition time. These data files are used to reproduce the test set infidelity evaluation and the two graphs shown in Fig 2.
Stored at the folder "Data_sets/Performance_evaluation"
Called by the codes:
- Main_results.ipynb (50 ms file only)
- Graph_Number_of_channels.ipynb (50 ms file only)
- Graph_Repetition_rate.ipynb (all ms versions)

### Dense_connective_tissue_data.npz
A data file containing measured count distributions for each pixel of the polarization scan of dense connective tissue, shown in Fig. 3.
Stored at the folder "Data_sets/Scan_Dense_connective_tissue"
Called by the codes:
- Dense_connective_tissue.ipynb

### Graph_Diatom.npz
A data file containing measured count distributions for each time bin of the polarization scan of a moving diatom, shown in Fig. 4.
Stored at the folder "Data_sets/Scan_Diatom"
Called by the codes:
- Dense_connective_tissue.ipynb

### Used_channel_combination.npy
A numpy array containing a list of active detection channels for each deep learning model used for reproducing the graph shown in the (a) panel of Fig. 2.
Stored at the main directory.
Called by the codes:
- Dense_connective_tissue.ipynb

















