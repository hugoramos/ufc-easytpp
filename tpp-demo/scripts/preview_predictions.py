# scripts/preview_predictions.py
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

# usa o bloco NHP_generate do YAML
cfg = Config.build_from_yaml_file('configs/taxi_nhp.yaml', experiment_id='NHP_generate')
runner = Runner.build_from_config(cfg)

# roda a geração e captura as saídas do runner (o TPPRunner salva/retorna batches)
outputs = runner.run()  # dependendo da versão, retorna None e apenas salva; tudo bem

# Exemplo: se preferir, rode eval e pegue batches do dataloader manualmente:
# (não é necessário se o seu runner já imprime / salva)
print("Geração finalizada. Veja os artefatos/logs em ./checkpoints/ e no console.")
