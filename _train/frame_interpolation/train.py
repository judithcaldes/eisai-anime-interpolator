



import sys
if sys.path[0]!='': sys.path.insert(0, '')

from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


ap = argparse.ArgumentParser()
ap.add_argument('dataset', type=str)
ap.add_argument('output', type=str)
args = ap.parse_args()


#################### train ####################

from _train.frame_interpolation.models.trainmodel import TrainModel
model = TrainModel()

# Load the pre-trained weights for SSL and DTM
model.ssl.load_state_dict(torch.load('./checkpoints/ssl.pt'))
model.dtm.load_state_dict(torch.load('./checkpoints/dtm.pt'))

from _train.frame_interpolation.datasets.rrldextr import Datamodule
dm = Datamodule(args.dataset, bs=4)

trainer = pl.Trainer(
    precision=16,
    max_epochs=400,
    gradient_clip_val=1.0,

    default_root_dir=mkdir(args.output),

    accelerator='ddp',
    gpus=1,
    accumulate_grad_batches=8,

    log_every_n_steps=10,
    
    callbacks=[pl.callbacks.ModelCheckpoint(
        monitor='val_lpips',
        mode='min',
        filename='{epoch:04d}-{val_lpips:0.6f}',
        save_top_k=8,
        dirpath=mkdir(f'{args.output}/checkpoints'),
        save_last=True,
    )],
    logger=[pl.loggers.TensorBoardLogger(
        mkdir(f'{args.output}/logs'),
        name='tensorboard',
        version=0,
        log_graph=False,
        default_hp_metric=True,
        prefix='',
    )],
    terminate_on_nan=True,
)

# Training wrapped in a try-except block to handle errors
try:
    trainer.fit(
        model,
        datamodule=dm,
    )
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")


torch.save(model.ssl.state_dict(), './checkpoints/ssl_judith.pt')
torch.save(model.dtm.state_dict(), './checkpoints/dtm_judith.pt')







