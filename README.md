# Trackmania

Wszystkie użyte biblioteki
```bash
pip install -r requirements.txt
```
Folder RL_Telemetry należy umieścić w folderze Plugins w OpenPlanet

Po zainicjowaniu środowiska Trackmania na docelowym torze, zweryfikuj czy działa nasłuch na porcie TCP:9000. Uruchom architekturę uczącą algorytmu SAC, wywołując poniższe polecenie:

```bash
python sac_learner.py
```

Hiperparametry modelu podlegają modyfikacji bezpośrednio w skrypcie wykonawczym. Wizualizację metryk treningowych w czasie rzeczywistym obsługuje TensorBoard, który zainicjujesz komendą:

```bash
tensorboard --logdir ./logs
```