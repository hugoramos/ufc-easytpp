import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/taxi_nhp.yaml')
    p.add_argument('--exp', default='NHP_train')
    args = p.parse_args()

    cfg = Config.build_from_yaml_file(args.config, experiment_id=args.exp)
    runner = Runner.build_from_config(cfg)
    runner.run()

if __name__ == '__main__':
    main()
