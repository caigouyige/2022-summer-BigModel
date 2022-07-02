import yaml
import sys

with open('config.yaml') as f:
    yaml_file = yaml.load(f, Loader=yaml.FullLoader)
  
yaml_file["train_batch_size"] = int(sys.argv[1])
yaml_file["learning_rate"] = float(sys.argv[2])
yaml_file["model_name"] = str(sys.argv[3])
yaml_file["model_dir"] = 'weights/' + str(yaml_file["model_name"]) + '_' + str(yaml_file["train_batch_size"]) + '_' + str(yaml_file["learning_rate"])

with open('config.yaml', 'w') as f:
    yaml.dump(yaml_file, f)