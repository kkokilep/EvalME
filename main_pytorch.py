import yaml
from parsing import get_args


def main(cfg):
    pass


if __name__ == "__main__":
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

    



