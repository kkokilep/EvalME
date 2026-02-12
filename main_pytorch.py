import yaml
from parsing import get_args
from pytorch import eval
import wandb
import os

# checking if the directory demo_folder 
# exist or not.

def main(cfg):
    
    if not os.path.exists("./wandb_logs"):
    
        os.makedirs("./wandb_logs")

    
    run = wandb.init(project="EvalME", name=cfg['experiment']['name'],dir="./wandb_logs")

    eval.evaluate(cfg)

if __name__ == "__main__":
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

    



