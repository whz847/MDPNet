## Training the model
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1026 train.py
