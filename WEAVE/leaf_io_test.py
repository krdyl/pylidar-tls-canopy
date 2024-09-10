import pandas as pd

hemi_file = '/mnt/c/users/kdayal/Downloads/LEAF_309_Tumba_DATA_manual_download_20230312/ESS00309_0018_hemi_20210706-110031Z_0200_0100.csv'



leaf_file = pd.read_csv(hemi_file, comment='#', na_values=-1.0,
            names=['sample_count','scan_encoder','rotary_encoder','range1',
                   'intensity1','range2','sample_time'], on_bad_lines='warn')