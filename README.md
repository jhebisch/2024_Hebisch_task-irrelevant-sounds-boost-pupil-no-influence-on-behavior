# 2024_Hebisch_Task-irrelevant-sounds-boost-pupil-no-influence-on-behavior
This project contains analysis code from the preprint: Hebisch, J., Ghassemieh, A. C., Zhecheva, E., Brouwer, M., van Gaal, S., Schwabe, L., Donner, T. H. & de Gee, J. W. (2024). Task-irrelevant stimuli reliably boost phasic pupil-linked arousal but do not affect decision formation. bioRxiv, 2024-05. https://doi.org/10.1101/2024.05.14.594080

The scripts preprocess and analyse behavioral and eye-tracking data of three perceptual decision-making experiments.

## Scripts
### Preprocessing scripts
0_preprocess: preprocessing of experiment 1 and 3 data
0_preprocess_hamburg_contrast_detection: preprocessing of experiment 2 data

### Analysis scripts
1_amsterdam_contrast_detection: Analysis of experiment 1 (contrast detection task conducted in Amsterdam)
2_hh_contrast_detection: Analysis of experiment 2 (contrast detectoin task conducted in Hamburg)
3_amsterdam_orientation_discrimination: Analysis of experiment 3 (orientation discrimination task conducted in Amsterdam)

### Helper scripts
utils: contains functions used in analysis or preprocessing of more than one experiment
utils_amsterdam_contrast_detection: contains functions used when dealing with experiment 1
utils_hh_contrast_detection: contains functions used when dealing with experiment 2
utils_amsterdam_orientation_discrimination: contains functions used when dealing with experiment 3

## Data
Data will be made available online.

To apply analysis scripts the following folder structure is needed

project_folder
    - analysis (with all scripts of this repository)
    - figs
    - data (folder with subfolders for each experiment, but also contains preprocessed dataframes of behavior and pupil epochs)
        - amsterdam_contrast_detection
        - hh_contrast_detection
        - amsterdam_orientation_discrimination

## References in Code
Pupil responses to blinks and saccades were corrected using a double gamma function convolution (Knapen et al., 2016)

For detecting blinks that were not detected by the Eyelink software, we use a custom algorithm that is courtesy of Ruud van den Brink (see van den Brink et al., 2016)

Knapen, T. et al. Cognitive and Ocular Factors Jointly Determine Pupil Responses under Equiluminance. PloS One 11, e0155574. https://doi.org/10.1371/journal.pone.0155574 (2016).

van den Brink, R. L., Murphy, P. R., & Nieuwenhuis, S. Pupil Diameter Tracks Lapses of Attention. PLOS ONE 11, e0165274. https://doi.org/10.1371/journal.pone.0165274 (2016).

_If you have any questions, open an issue or get in touch @JosefineHebisch_