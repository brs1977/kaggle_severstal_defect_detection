# flake8: noqa
from catalyst.dl import registry, SupervisedRunner as Runner
from experiment import Experiment
from mlcomp.contrib.catalyst import register
import segmentation_models_pytorch as smp 

import warnings
warnings.filterwarnings("ignore")

# registry.Model(smp.Unet, name='SMPUnet')
registry.Model(smp.FPN, name='SMPFPN')
register()
