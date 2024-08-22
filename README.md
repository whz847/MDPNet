This project is developed based on the DDP framework, and the code will be updated later.
## Training the model
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1026 train.py
