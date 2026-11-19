Experimental data for the hands-on notebook (not included in the repository)
============================================================================

Put the three downloaded data files in this folder (data/experimental/):

    dataset_1.mat     (44 MB)
    dataset_2.mat     (45 MB)
    dataset_3.mat     ( 3 MB)

They are in the course Dropbox folder (motility_module/data). The link is announced
on the Canvas course website and restated in class. Download them, then drag them
into this folder. The notebook
notebooks/hands_on_real_tracks.ipynb lists what it finds here and stops with a message
if the folder is empty. You choose there which dataset to analyze.

What the files hold: tracked swimming bacteria from three experiments, two species
in two environments (which ones: see Canvas and the class notes). Each file has one MATLAB variable,
`tracks`: a struct array with one element per cell and three fields, x and y
(positions in micrometers) and time (seconds), each with one value per frame.
NaN marks a frame in which the tracker lost the cell.

Git ignores every file in this folder except this README, so the data are never
committed by accident.
