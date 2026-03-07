# Plan - Praca Magisterska

## Temat pracy

Analiza porównawcza paradygmatów trenowania agentów autonomicznych w Trackmania: wpływ pojemności sieci neuronowej i uczenia sekwencyjnego vs. wielozadaniowego na katastrofalne zapominanie.

Badanie implementacyjne - cel: wyprodukować dane empiryczne porównujące:
- Uczenie sekwencyjne (track1 -> track2 -> track3) vs. wielozadaniowe (mieszane)
- Rozmiary sieci (Tiny 25K, Small 150K, Large 1.2M, Huge 4M parametrów)
- Miernik: Catastrophic Forgetting = wydajność_przed - wydajność_po

---

## Etap 1: Debugowanie i naprawa (PRIORYTET ZERO)

### Krok 1: Napraw Exit Code 1 w sac_learner.py

Uruchom:
  python -u sac_learner.py 2>&1 | tee debug.log

Szukaj w logu:
- ImportError (TMRL? PyGame? Stable-baselines3?)
- CUDA error (czy torch.cuda.is_available()?)
- Multiprocessing error (spawn vs fork context)

Jeśli import TMRL - sprawdź:
  python -c "from tmrl.actor import TorchActorModule"

Jeśli CUDA - sprawdź:
  python -c "import torch; print(torch.cuda.is_available())"

Czas: 2-4 godziny

### Krok 2: Dodaj metrykę Catastrophic Forgetting

W trackmania_env.py dodaj:
- tracked_rewards = {} # słownik Track -> lista nagrodzień
- W metodzie step(): track_id -> add reward to tracked_rewards[track_id]
- Before switching track: save avg reward as R_before
- After training on new track: compute R_after = current avg
- Log CF = R_before - R_after do TensorBoard

Czas: 2 godziny

### Krok 3: Włącz checkpointing

W sac_learner.py:
- RESUME_TRAINING = True
- Na starcie load latest model if exists

W tmrl_sac_agent.py:
- Napraw metodę load() aby prawidłowo wczytywać SAC model

Czas: 1.5 godziny

### Krok 4: Supervisor procesów

Zamiast bezpośrednio startować 3 workerów, dodaj loop:
- Jeśli process.is_alive() == False, uruchom na nowo
- Loguj restart events
- Umożliwi 48+ godzinowe trenowanie

Czas: 2 godziny

---

## Etap 2: Architektura i warianty sieci (po naprawie)

### Krok 5: Stwórz 4 warianty sieci - ArchitectureFactory

W tmrl_sac_agent.py dodaj:

  class ArchitectureFactory:
    @staticmethod
    def create_tiny():
      return MLPPolicy([128, 256])  # 25K params
    
    @staticmethod
    def create_small():
      return MLPPolicy([256, 512])  # 150K params
    
    @staticmethod
    def create_large():
      return MLPPolicy([512, 1024]) # 1.2M params
    
    @staticmethod
    def create_huge():
      return MLPPolicy([1024, 2048]) # 4M params

Config: ARCHITECTURE = "tiny|small|large|huge" w sac_learner.py

Test: Każdy wariant musi wytrenować co najmniej 100 batchów bez crasha

Czas: 3 godziny

### Krok 6: CNN encoder do wizji

Zamiast 49159D flat, zrób:

  class CNNVisionEncoder(nn.Module):
    Vision 128x128x3:
      Conv2D(32, kernel=3) + ReLU + MaxPool2D
      Conv2D(64, kernel=3) + ReLU + MaxPool2D
      Flatten -> 4096D
    
    Telemetry 7D:
      Linear(32)
    
    Concatenate: 4128D total -> SAC Policy

Impact: 15-20% szybsze trenowanie

Czas: 4 godziny

---

## Etap 3: Eksperymenty (Phase 3)

Testuj kombinacje: 4 architektury x 3 paradygmaty = 12 eksperymentów

Warianty:
- Tiny (25K)
- Small (150K)
- Large (1.2M)
- Huge (4M)

Paradygmaty:
- Sequential: Track1 (6h) -> Track2 (6h) -> Track3 (6h)
- Multi-Task: Rotuj Tracki co 2 godziny przez 12h
- Specialist: 3 osobne agenty, każdy tylko 1 track

### Krok 7: Uczenie sekwencyjne

Dla każdego wariantu (tiny/small/large/huge):
1. Trenuj 6 godzin na Track1
2. Zaloguj performance na Track1 -> R_1_before
3. Trenuj 6 godzin na Track2
4. Zaloguj performance na Track1 i Track2 -> calculate CF
5. Trenuj 6 godzin na Track3
6. Zaloguj performance na Track1, Track2, Track3 -> calculate CF

Razem: 18h x 4 architektury x 2 trials = 144 godziny

Loguj wszystko do TensorBoard z tagiem "sequential_{arch}"

### Krok 8: Wielozadaniowe

Dla każdego wariantu:
1. Stwórz curriculum: rotacja Track1 -> Track2 -> Track3 co 2 godziny
2. Trenuj przez 12 godzin z rotacją
3. Za każdą godzinę zaloguj performance na KAŻDYM tracku jednocześnie

Razem: 12h x 4 architektury x 2 trials = 96 godzin

Loguj wszystko z tagiem "multitask_{arch}"

### Krok 9: Specjaliści

Paralelnie (jeśli multi-GPU):
1. Agent 1 <- Train Track1 przez 6h
2. Agent 2 <- Train Track2 przez 6h
3. Agent 3 <- Train Track3 przez 6h

Razem: 6h x 3 = 18 godzin

Porównaj performance specjalistów vs. generalisty (multi-task)

### Co logować w każdym eksperymencie:

reward_moving_avg (co 10 epizodów)
policy_loss
value_loss
episodes_to_threshold
per_track_performance
model_parameter_count
training_time_per_episode

Wszystko trafić do jednego TensorBoard, podzielić po tag: "sequential_tiny", "multitask_large" etc

---

## Etap 4: Analiza i pisanie pracy

### Krok 10: Exportuj dane z TensorBoard

Dla każdego eksperymentu:
1. Odczytaj .tfevents fajla z logs/ folderu
2. Stwórz CSV: timestamp, step, reward, loss, track_id, architecture
3. Oblicz statystyki: mean, std, min, max dla każdego wariantu

Czas: 2 godziny

### Krok 11: Analiza statystyczna

Dla każdej pary danych:

ANOVA na effect architecture:
- H0: rozmiar sieci nie wpływa na CF
- Test: F-statistic, p-value < 0.05

T-test sequential vs multitask:
- H0: brak różnicy w CF między paradygmatami
- Test: mean CF_seq vs CF_multi, p-value < 0.05

Korelacja: parameter_count vs catastrophic_forgetting
- Pearson correlation + significance

Output: tabela wyników z p-values

Czas: 3 godziny

### Krok 12: Wykresy i wizualizacje

Figura 1: Krzywe konwergencji
- X: epizody, Y: reward
- Linia dla każdej architektury
- Osobne subploty dla sequential/multitask/specialist

Figura 2: Heatmapa CF
- X: przejście (Track1->Track2, itd)
- Y: architektura (tiny/small/large/huge)
- Kolor: wartość CF

Figura 3: Porównanie specjalista vs generalist
- Bar chart: średnia reward specjalisty vs generalisty
- Pokazać confidence intervals

Figura 4: Wpływ rozmiaru sieci na CF
- Scatter: parameter_count vs CF
- Linia trendu

Czas: 3 godziny

### Krok 13: Sekcje pracy

Sekcja 1: Wstęp (3-4 strony)
- Problem: catastrophic forgetting w sterowaniu autonomicznym
- Pytania badawcze
- Cele

Sekcja 2: Przegląd literatury (8-10 stron)
- SAC, PPO, DQN
- EWC, Experience Replay
- Scaling laws sieci
- Trackmania + RL

Sekcja 3: Metodologia (6-8 stron)
- Specyfikacja Trackmania env
- Definicja reward function
- Space obserwacji i akcji
- SAC parametry
- Definicja metryk CF i transferu

Sekcja 4: Wyniki (12-15 stron)
- Krzywe konwergencji z errorbarami
- Analiza CF vs architektura
- Analiza CF vs paradygmat
- Specjalista vs generalist
- Transfer learning efficiency

Sekcja 5: Dyskusja i wnioski (8-10 stron)
- Intepretacja wyników
- Odpowiedź na 5 pytań badawczych
- Praktyczne implikacje
- Future work
- References (30-40)

---

## Timeline

Tydzień 1: Debugowanie + CF metric
- Napraw Exit Code 1
- Dodaj CF miernik
- Włącz checkpointing
- Supervisor procesów

Tydzień 2-3: Architektury
- ArchitectureFactory (4 warianty)
- CNN encoder
- Config management

Tydzień 4-5: Uczenie sekwencyjne
- 4 architektury x 2 trials = 8 runs
- 18h x 8 = 144 godzin

Tydzień 5-6: Wielozadaniowe
- 4 architektury x 2 trials = 8 runs
- 12h x 8 = 96 godzin

Tydzień 6-7: Specjaliści
- 3 tracki x 2 trials = 6 runs
- 6h x 6 = 36 godzin

Tydzień 8: Analiza
- Export danych
- Statystyka
- Wykresy

Tydzień 8-9: Pisanie
- Sekcje 1-3
- Draft rezultatów

Tydzień 9-10: Finalizacja
- Sekcje 4-5
- References
- Przygotowanie obrony

---

## Kryteria sukcesu

Po Etapie 1 (1 tydzień):
- sac_learner.py uruchamia się bez Exit Code 1
- TensorBoard zbiera dane co najmniej przez 2 godziny
- CF metric pojawia się w logach

Po Etapie 2 (3 tygodnie):
- 4 architektur trenuje się bez crash
- CNN encoder działa
- Performance różni się między wariantami (co najmniej 10% min/max)

Po Etapie 3 (7 tygodni):
- 24-36 eksperymentów ukończonych
- Wszystkie dane w TensorBoard
- ANOVA wykazuje p < 0.05 dla effect architecture
- Różnica sequential vs multitask > 5%

Po Etapie 4 (10 tygodni):
- Praca 50+ stron
- 5 figur głównych
- Odpowiedź na 5 pytań badawczych
- Gotowy do obrony

---

## 5 pytań badawczych do odpowiedzi

Q1: Czy rozmiar sieci wpływa na catastrophic forgetting?

Q2: Czy uczenie sekwencyjne powoduje większe forgetting niż wielozadaniowe?

Q3: Jaka jest cena uniwersalności - różnica performance między specjalistą a generalistą?

Q4: Jak skaluje się efektywność transferu w zależności od rozmiaru sieci?

Q5: Jaki jest optymalny rozmiar sieci do sterownika w Trackmanią?

---

Ostatnia aktualizacja: 2026-03-07