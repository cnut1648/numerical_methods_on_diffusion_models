# python run.py train=optim
# python run.py train=optim dataset.dataset=KMNIST
python run.py train=optim dataset.dataset=FMNIST
python run.py train=optim dataset.dataset=FMNIST model=PF train.batch_size=1024
python run.py train=optim dataset.dataset=KMNIST model=PF train.batch_size=1024
python run.py train=optim model=PF train.batch_size=1024