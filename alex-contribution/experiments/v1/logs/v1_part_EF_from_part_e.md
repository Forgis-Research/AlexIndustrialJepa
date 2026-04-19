# v1 Phase 3 (Part E direct) — 2026-04-19 19:26

Loading FD001 for Phase 3 (fine-tune)...

============================================================
PART E: Fine-tuning at Multiple Label Budgets
============================================================

  --- Label budget: 100% ---
    Using 85 training engines
    LSTM seed 0: test RMSE = 17.75
    LSTM seed 1: test RMSE = 15.80
    LSTM seed 2: test RMSE = 19.09
    LSTM seed 3: test RMSE = 18.07
    LSTM seed 4: test RMSE = 16.10
    Supervised LSTM: 17.36 +/- 1.24
    JEPA frozen seed 0: test RMSE = 20.66
    JEPA frozen seed 1: test RMSE = 20.53
    JEPA frozen seed 2: test RMSE = 20.69
    JEPA frozen seed 3: test RMSE = 21.16
    JEPA frozen seed 4: test RMSE = 21.00
    JEPA frozen: 20.81 +/- 0.23
    JEPA E2E seed 0: test RMSE = 15.62
    JEPA E2E seed 1: test RMSE = 15.42
    JEPA E2E seed 2: test RMSE = 14.38
    JEPA E2E seed 3: test RMSE = 14.74
    JEPA E2E seed 4: test RMSE = 16.65
    JEPA E2E: 15.36 +/- 0.78

  --- Label budget: 50% ---
    Using 42 training engines
    LSTM seed 0: test RMSE = 17.69
    LSTM seed 1: test RMSE = 17.86
    LSTM seed 2: test RMSE = 18.78
    LSTM seed 3: test RMSE = 19.54
    LSTM seed 4: test RMSE = 17.62
    Supervised LSTM: 18.30 +/- 0.75
    JEPA frozen seed 0: test RMSE = 21.02
    JEPA frozen seed 1: test RMSE = 20.70
    JEPA frozen seed 2: test RMSE = 20.53
    JEPA frozen seed 3: test RMSE = 20.62
    JEPA frozen seed 4: test RMSE = 20.46
    JEPA frozen: 20.67 +/- 0.20
    JEPA E2E seed 0: test RMSE = 18.13
    JEPA E2E seed 1: test RMSE = 16.33
    JEPA E2E seed 2: test RMSE = 17.68
    JEPA E2E seed 3: test RMSE = 15.20
    JEPA E2E seed 4: test RMSE = 15.17
    JEPA E2E: 16.50 +/- 1.23

  --- Label budget: 20% ---
    Using 17 training engines
    LSTM seed 0: test RMSE = 18.25
    LSTM seed 1: test RMSE = 19.07
    LSTM seed 2: test RMSE = 19.75
    LSTM seed 3: test RMSE = 17.37
    LSTM seed 4: test RMSE = 18.33
    Supervised LSTM: 18.55 +/- 0.81
    JEPA frozen seed 0: test RMSE = 21.04
    JEPA frozen seed 1: test RMSE = 20.79
    JEPA frozen seed 2: test RMSE = 20.48
    JEPA frozen seed 3: test RMSE = 21.12
    JEPA frozen seed 4: test RMSE = 20.52
    JEPA frozen: 20.79 +/- 0.26
    JEPA E2E seed 0: test RMSE = 17.69
    JEPA E2E seed 1: test RMSE = 17.68
    JEPA E2E seed 2: test RMSE = 16.00
    JEPA E2E seed 3: test RMSE = 14.78
    JEPA E2E seed 4: test RMSE = 16.81
    JEPA E2E: 16.59 +/- 1.10

  --- Label budget: 10% ---
    Using 8 training engines
    LSTM seed 0: test RMSE = 20.59
    LSTM seed 1: test RMSE = 22.63
    LSTM seed 2: test RMSE = 45.01
    LSTM seed 3: test RMSE = 44.07
    LSTM seed 4: test RMSE = 23.79
    Supervised LSTM: 31.22 +/- 10.93
    JEPA frozen seed 0: test RMSE = 23.11
    JEPA frozen seed 1: test RMSE = 24.99
    JEPA frozen seed 2: test RMSE = 21.01
    JEPA frozen seed 3: test RMSE = 21.58
    JEPA frozen seed 4: test RMSE = 22.67
    JEPA frozen: 22.67 +/- 1.38
    JEPA E2E seed 0: test RMSE = 28.29
    JEPA E2E seed 1: test RMSE = 31.23
    JEPA E2E seed 2: test RMSE = 25.64
    JEPA E2E seed 3: test RMSE = 21.24
    JEPA E2E seed 4: test RMSE = 23.05
    JEPA E2E: 25.89 +/- 3.58

  --- Label budget: 5% ---
    Using 4 training engines
    LSTM seed 0: test RMSE = 27.62
    LSTM seed 1: test RMSE = 26.02
    LSTM seed 2: test RMSE = 45.16
    LSTM seed 3: test RMSE = 44.23
    LSTM seed 4: test RMSE = 22.36
    Supervised LSTM: 33.08 +/- 9.64
    JEPA frozen seed 0: test RMSE = 23.39
    JEPA frozen seed 1: test RMSE = 20.89
    JEPA frozen seed 2: test RMSE = 20.75
    JEPA frozen seed 3: test RMSE = 24.08
    JEPA frozen seed 4: test RMSE = 20.89
    JEPA frozen: 22.00 +/- 1.43
    JEPA E2E seed 0: test RMSE = 22.95
    JEPA E2E seed 1: test RMSE = 23.67
    JEPA E2E seed 2: test RMSE = 27.59
    JEPA E2E seed 3: test RMSE = 23.25
    JEPA E2E seed 4: test RMSE = 22.30
    JEPA E2E: 23.95 +/- 1.87

[alex-wrapper] Part E wall time: 6.66 min
  Saved /home/sagemaker-user/AlexIndustrialJepa/alex-contribution/experiments/v1/plots/v1_label_efficiency.png

[alex-wrapper] Fine-tune summary (my v1 reproduction):
  supervised_lstm    @ 100%: 17.36 +/- 1.24
  supervised_lstm    @  50%: 18.30 +/- 0.75
  supervised_lstm    @  20%: 18.55 +/- 0.81
  supervised_lstm    @  10%: 31.22 +/- 10.93
  supervised_lstm    @   5%: 33.08 +/- 9.64
  jepa_frozen        @ 100%: 20.81 +/- 0.23
  jepa_frozen        @  50%: 20.67 +/- 0.20
  jepa_frozen        @  20%: 20.79 +/- 0.26
  jepa_frozen        @  10%: 22.67 +/- 1.38
  jepa_frozen        @   5%: 22.00 +/- 1.43
  jepa_e2e           @ 100%: 15.36 +/- 0.78
  jepa_e2e           @  50%: 16.50 +/- 1.23
  jepa_e2e           @  20%: 16.59 +/- 1.10
  jepa_e2e           @  10%: 25.89 +/- 3.58
  jepa_e2e           @   5%: 23.95 +/- 1.87
