from ray.rllib.agents.ppo import PPOTrainer

# Configure the algorithm.
iterations = 10
config = {
    "env": "MountainCarContinuous-v0",
    "num_gpus": 0,
    "num_workers": 2,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
        "use_lstm":True,
    },
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": False,
    },
}

trainer = PPOTrainer(config=config)
print("Starting training")
for i in range(iterations):
    print(f"Iteration %d" % i)
    print(trainer.train())
print("Evaluating")
print(trainer.evaluate())
print("Done")
