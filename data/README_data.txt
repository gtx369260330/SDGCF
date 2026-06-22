Place processed three-channel Sleep-EDF EEG/EOG data in:

    data/data_multimodal_eeg_eog_3ch

Required files:

    X_all.npy              shape [N, 3, 3000]
    y_all.npy              shape [N], labels 0-4
    subject_id_all.npy     shape [N], used for subject-level splitting

Optional files:

    record_id_all.npy
    dataset_all.npy
    channel_info.csv
    label_map.csv

An .npz file can also be passed directly:

    python run_all_experiments.py --data_path your_data.npz --root_dir all_experiment_results

Supported .npz keys:

    X_all / y_all / subject_id_all
    X / y / subjects
