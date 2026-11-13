<div align="center">

# 🏎️ BOX, BOX

### FORMULA 1 · DATA ANALYTICS · RACE STRATEGY

**Behind every lap is a story. Behind every second is data.**

An exploratory data science project investigating driver performance, tyre degradation, pit-stop strategy and qualifying-to-race relationships through lap-by-lap Formula 1 data.

<br>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org/)

<br>

[📊 Explore Analysis](notebooks/race_analysis.ipynb) &nbsp; • &nbsp;
[📁 Project Structure](#-project-architecture) &nbsp; • &nbsp;
[⚙️ Installation](#-getting-started) &nbsp; • &nbsp;
[📈 Key Findings](#-race-insights)

</div>

---

## 🏁 The Project

Formula 1 is a sport where milliseconds can determine the outcome of an entire race.

But raw lap times alone do not tell the complete story.

**BOX, BOX** uses Python and statistical methods to investigate the 2026 British Grand Prix, examining race pace, driver consistency, tyre degradation and strategic decisions.

The project analyses both qualifying and race-session data to understand how performance evolves throughout a Grand Prix.

<table>
<tr>
<td width="33%" align="center">

### 01

**DRIVER PERFORMANCE**

Who demonstrated the strongest race pace and consistency?

</td>
<td width="33%" align="center">

### 02

**TYRE & STRATEGY**

How did tyre degradation and pit-stop strategies influence performance?

</td>
<td width="33%" align="center">

### 03

**QUALIFYING vs RACE**

How strongly did qualifying performance predict race pace?

</td>
</tr>
</table>

---

## 📊 Race Insights

A snapshot of the statistical findings from our analysis.

<table>
<tr>
<td align="center" width="25%">

### 0.95

**Spearman's ρ**

Qualifying vs race-pace rank

</td>
<td align="center" width="25%">

### 0.44%

**Consistency**

Norris — lowest variability

</td>
<td align="center" width="25%">

### 0.05–0.06s

**Tyre Degradation**

Estimated loss per lap of tyre age

</td>
<td align="center" width="25%">

### 30s

**Pit-Lane Time**

Approximate green-flag pit stop

</td>
</tr>
</table>

### 🏆 Driver Performance

| Finding | Result |
|---|---|
| Best median race pace | Charles Leclerc |
| Second-best race pace | Kimi Antonelli |
| Third-best race pace | Max Verstappen |
| Most consistent driver | Lando Norris |
| Fastest individual lap | Kimi Antonelli — 91.777s |

**Key observation:** Antonelli recorded the fastest individual lap but did not have the fastest median race pace. A single fast lap is not necessarily representative of sustained race performance.

### 🛞 Tyre Degradation

- Medium and hard compounds showed approximately 0.05–0.06 seconds of additional lap time per lap of tyre age.
- The difference between the compounds was not statistically significant (Mann–Whitney U, p = 0.13).
- Soft-tyre data was insufficient for a reliable comparison.

### 📉 Fuel Correction

Fuel correction reduced the average lap-time standard deviation from **0.93 seconds to 0.66 seconds**, helping isolate driver performance from the estimated effect of fuel consumption.

---

## 🔬 Analytical Methodology

The project follows a modular data science workflow.

<details>
<summary><b>01 — Data acquisition and preprocessing</b></summary>

- Load qualifying and race-session JSON files.
- Convert the column-oriented JSON structure into DataFrames.
- Handle missing values represented as the string `"None"`.
- Convert missing values into proper numerical NaN values.
- Decode track status information.
- Identify pit-in, pit-out and neutralised laps.

</details>

<details>
<summary><b>02 — Driver performance analysis</b></summary>

- Calculate median race pace.
- Measure lap-time consistency using standard deviation.
- Apply fuel-load correction.
- Compare individual driver performance.
- Calculate qualifying and race-pace rankings.

</details>

<details>
<summary><b>03 — Tyre degradation and strategy</b></summary>

- Identify individual tyre stints.
- Calculate tyre age.
- Estimate degradation using linear regression.
- Compare medium and hard compounds.
- Apply the Mann–Whitney U test.
- Estimate pit-lane time.

</details>

<details>
<summary><b>04 — Statistical analysis</b></summary>

- Descriptive statistics.
- Standard deviation and coefficient of variation.
- Linear regression.
- Mann–Whitney U hypothesis testing.
- Spearman rank correlation.

</details>

---

## 📈 Race Strategy: An Interesting Case

### Kimi Antonelli — Pole Position to P9

Antonelli started from pole position and demonstrated the second-best median race pace.

However, he finished ninth after three stops within nine laps (laps 35, 41 and 43).

This demonstrates an important distinction:

**Race performance is not determined by pace alone.**

Strategy, pit stops and race circumstances can significantly influence the final result.

The available dataset establishes what happened, but does not establish the reasons behind every strategic decision.

---

## 📂 Data Sources

The project uses publicly available Formula 1 timing data from:

**[TracingInsights — 2026 Repository](https://github.com/TracingInsights/2026)**

The required JSON files are stored locally, allowing the analysis to run without additional API requests.

| Dataset | Information |
|---|---|
| `race/session_laptimes.json` | Race lap times, positions, tyres, pit stops and track status |
| `race/drivers.json` | Driver identities, teams and team colours |
| `qualifying/session_laptimes.json` | Q1, Q2 and Q3 lap times |

---

## 🧱 Project Architecture

The project separates data processing, statistical analysis and visualization into independent Python modules.

```text
BOX-BOX-F1-ANALYTICS/
│
├── data/
│   ├── raw/
│   │   ├── qualifying/
│   │   │   └── session_laptimes.json
│   │   │
│   │   └── race/
│   │       ├── drivers.json
│   │       └── session_laptimes.json
│   │
│   └── processed/
│       ├── qualifying_laps_clean.csv
│       ├── qualifying_vs_race.csv
│       ├── race_laps_clean.csv
│       └── race_pace_summary.csv
│
├── notebooks/
│   └── race_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── driver_analysis.py
│   ├── tyre_analysis.py
│   └── visualisation.py
│
├── outputs/
│   └── plots/
│       ├── 1_pace_ranking.png
│       ├── 2_pace_vs_consistency.png
│       ├── 3_strategy.png
│       ├── 4_degradation.png
│       ├── 5_position_trace.png
│       └── 6_quali_vs_race.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Getting Started

### Requirements

- Python 3.11 or newer
- Git
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<https://github.com/jatin-satuse-datascience>/BOX-BOX-F1-ANALYTICS.git

cd BOX-BOX-F1-ANALYTICS

python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the analysis

```bash
jupyter lab
```

Open:

```text
notebooks/race_analysis.ipynb
```

Run all cells to reproduce the analysis.

Generated outputs are saved in `data/processed/` and `outputs/plots/`.

---

## ⚠️ Limitations

<details>
<summary><b>Data and methodological limitations</b></summary>

1. **Single-race analysis:** Results represent the 2026 British Grand Prix and cannot be generalised to an entire season.

2. **Fuel estimation:** Fuel correction relies on assumed initial fuel load and consumption rather than actual fuel telemetry.

3. **Tyre sample size:** Soft compounds have insufficient observations for reliable degradation estimates.

4. **Track evolution:** Changes in track conditions and traffic can influence lap-time trends.

5. **Qualifying conditions:** Drivers eliminated in Q1 or Q2 may have set their fastest laps under different track conditions from Q3 participants.

6. **Strategic interpretation:** The dataset records race events but does not establish the reasons behind every team decision.

</details>

---

## 🎯 Project Objective

This project demonstrates the application of Python programming, data manipulation, statistical inference and visualization to a real-world sporting dataset.

Rather than simply presenting race statistics, it aims to extract meaningful insights from performance data using reproducible analytical methods.

<div align="center">

### LIGHTS OUT. DATA ON. 🏁

**BOX, BOX — Formula 1 Data Analytics**

*Built with Python, statistics and a passion for motorsport.*

</div>
