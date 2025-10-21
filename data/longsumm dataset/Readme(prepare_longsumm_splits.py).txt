Run this prepare_longsumm_splits.py with the LongSumm-master-github to automatically createlongsumm_prepared.


Command:
python3 prepare_longsumm_splits.py --longsumm_root ./LongSumm-master-github --out_dir ./longsumm_prepared --seed 33


Next:
python3 make_scibart_pairs.py --longsumm_root "./LongSumm-master-github" --out_dir "./longsumm_prepared"