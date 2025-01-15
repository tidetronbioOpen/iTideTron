"""
Filename: main.py
Author: KONGFOO 
Contact: dev@kongfoo.cn
"""

import os, argparse
import logging, wandb
from datetime import datetime
from models.itidetron import ITidetron 
from version import __version__
  
log_format = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(filename=f'logs/run_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.log',
                     level=logging.INFO, format=log_format)

os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"
            
if __name__ == '__main__':
    """
    Main entrance to train or predict using the Model class.
    
    Usage: python main.py --config <config_file_path>
    
    Arguments:
        --config (str): Path to the configuration file containing model settings.
        --wandb (str): Name of the weight and bias project for logging experiment results if applicable.

    The script reads the configuration file, creates an instance of the Model class,
    and performs either training or prediction based on the provided status in the
    configuration file.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='Configuration File', default='config/debug.yml' )
    parser.add_argument('--wandb', help='wandb project name', default=None)
    args = parser.parse_args()
    if args.wandb: wandb.init(project=args.wandb, config={'version': __version__})
    iTidetron = ITidetron(args.config, __version__)
    if iTidetron.status.lower()=='train':
        iTidetron.train()
    elif iTidetron.status.lower()=='predict':
        iTidetron.predict()
    if wandb.run: wandb.finish()
