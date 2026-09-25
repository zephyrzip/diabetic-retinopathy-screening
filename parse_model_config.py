import json

config_path = r"d:\New folder (2)\models\best_dr_model_latest.keras\config.json"

with open(config_path, "r") as f:
    cfg = json.load(f)

print("Class name:", cfg.get("class_name"))
c = cfg.get("config", {})
print("Model name:", c.get("name"))
print("Input layers:", c.get("input_layers"))
print("Output layers:", c.get("output_layers"))

layers = c.get("layers", [])
print(f"Total layers: {len(layers)}")
for i, l in enumerate(layers):
    name = l.get("name")
    cname = l.get("class_name")
    cfg_layer = l.get("config", {})
    units = cfg_layer.get("units")
    activation = cfg_layer.get("activation")
    batch_shape = cfg_layer.get("batch_shape")
    inbound = l.get("inbound_nodes")
    print(f"[{i}] {name} ({cname}) | units: {units} | act: {activation} | batch_shape: {batch_shape}")
