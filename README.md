```
python -m pysc2.bin.agent --map MoveToBeacon --agent baleian.sc2.agents.MoveToBeaconSimpleAgent
```

```
python -m baleian.sc2.bin.play --env baleain.sc2.envs.MoveToBeacon --agent baleian.sc2.agents.MoveToBeaconSimpleAgent
```


```
python -m baleian.sc2.bin.train \
--env baleain.sc2.envs.MoveToBeacon \
--agent baleian.sc2.agents.MoveToBeaconDQNAgent \
--model-dir {model_path}
```

```
python -m baleian.sc2.bin.play \
--env baleain.sc2.envs.MoveToBeacon \
--agent baleian.sc2.agents.MoveToBeaconDQNAgent \
--model-dir {model_path}
```
