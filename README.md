# Trackmania

All required libraries
```bash
pip install -r requirements.txt
```
Place the RL_Telemetry folder in the Plugins folder in OpenPlanet

After initializing the Trackmania environment on your target track, verify that TCP port 9000 listening is working. Run the SAC algorithm training architecture with the following command:

```bash
python sac_learner.py
```

Model hyperparameters can be modified directly in the execution script. TensorBoard handles real-time training metrics visualization, initialized with:

```bash
tensorboard --logdir ./logs
```