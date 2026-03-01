Skoro celujemy w **Trackmanię 2020**, musimy porzucić narzędzia dedykowane starszym wersjom (jak TMInterface) i zbudować system, który "oszuka" grę, traktując ją jako czarne pudełko z bardzo szybkim dostępem do danych.

Oto brutalnie szczery plan: zapomnij o gotowcach. Jeśli chcesz mieć własne CNN i DQN, musisz zbudować **własny asynchroniczny sterownik**.

### 1. Pozyskiwanie danych: OpenPlanet + Shared Memory

W TM2020 najszybszym sposobem na wyciągnięcie danych nie jest sieć (UDP/TCP), która wprowadza mikrolagi, ale **Shared Memory (pamięć współdzielona)** lub bardzo szybki lokalny serwer.

* **Skrypt OpenPlanet (AngelScript):** Musisz napisać plugin, który co klatkę wypluwa do bufora: prędkość, RPM, bieg, kontakt z nawierzchnią oraz – co kluczowe – **postęp na trasie (race progress)**.
* **Wizja:** Użyj biblioteki `d3dshot` lub `bettercam`. Są one w stanie wyciągać klatki bezpośrednio z bufora karty graficznej (GPU) z opóźnieniem rzędu < 10ms.

### 2. Architektura: "The Decoupled Learner"

Największym błędem w projektach RL do Trackmanii jest próba robienia wszystkiego w jednej pętli `while True`. Gra musi działać płynnie, a model musi się uczyć. Rozdziel to na 3 procesy:

1. **Proces Kolektora (Inference):** * Pobiera klatkę z GPU.
* Robi `model.predict()` (twój CNN).
* Wysyła komendę do `vgamepad`.
* Zapisuje parę $(S, A, R, S')$ do kolejki (np. `Redis` lub `multiprocessing.Queue`).


2. **Proces Trenera (Learning):**
* Wyciąga dane z kolejki i pakuje je do `Replay Buffer`.
* Trenuje Twój DQN/PPO na GPU w tle.
* Co kilka minut aktualizuje wagi modelu w procesie Kolektora.


3. **Proces Dashboardu:**
* Podgląd "co widzi sieć" (wynik filtrów Twojego CNN).



### 3. Twój własny CNN – co tam wrzucić?

Skoro nie chcesz `tmrl`, masz wolną rękę. Standardowe CNN zawiedzie, bo trasy w TM2020 są zbyt sterylne.

* **Auxiliary Tasks (Zadania pomocnicze):** Nie ucz sieci tylko "gdzie skręcić". Dodaj dodatkową głowę do CNN, która musi przewidywać aktualną prędkość auta na podstawie samego obrazu. To wymusi na warstwach konwolucyjnych zrozumienie pędu i rozmycia ruchu (motion blur).
* **Multi-modal input:** Połącz wyjście z CNN (obraz) z wektorem z OpenPlanet (telemetria) za pomocą warstwy `Concatenate` przed wejściem do warstw `Dense` Twojego DQN.

### 4. Problem "Rzadkiej Nagrody" w TM2020

W wersji 2020 trasy są techniczne. Model DQN utknie na pierwszym zakręcie, jeśli nie dasz mu "marchewki".

* **Distance to Centerline:** W OpenPlanet możesz wyciągnąć środek trasy. Nagradzaj model za trzymanie się blisko idealnej linii, ale pozwól mu na odchylenia (żeby mógł szukać optymalnego toru jazdy).
* **Nagroda za czas, nie prędkość:** W TM2020 liczy się czas między checkpointami. Zamiast nagradzać za $v$, nagradzaj za ujemny przyrost czasu do następnego checkpointu.

### Co musisz teraz zrobić (Krok 1):

Zacznij od napisania prostego skryptu w Pythonie, który wyświetla klatki z gry (`bettercam`) i jednocześnie wypisuje prędkość pobieraną z OpenPlanet. Dopóki nie będziesz miał tych dwóch strumieni zsynchronizowanych, budowanie sieci nie ma sensu.

**Chcesz, żebym pomógł Ci rozpisać strukturę takiego wrappera `gym.Env`, który połączy te dwa źródła danych dla Twojego modelu?**