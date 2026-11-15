# Weather Guru: Trend Detective

**Is the change in NASA Earth data real, or just luck?**

Built for the 2026 NASA Space Apps Challenge (Earth Science and Software).

- **Live demo:** [weather-guru.streamlit.app](https://weather-guru.streamlit.app/)
- **Demo video (30 seconds):** [Watch on Google Drive](https://drive.google.com/file/d/1g-4diFvyEBYZCxngNsMi6YiIoSpywnSE/view?usp=sharing)

---

## Why we built this

Karachi feels hotter every year, and people say it rains less. Lahore and Delhi sit among the most polluted cities in the world every winter. Everyone has a theory: factories, traffic, concrete, crop burning.

We wanted to check those theories against real satellite and model data instead of arguing about them. A graph on its own can't do that. Seasons hide slow change, and random noise can look like a trend. So Trend Detective doesn't stop at the graph. It tells you how much something changed, and whether that change is real or could just be luck.

## What it does

You follow five steps, the same ones the challenge describes:

1. **Pick a variable**, like temperature, rainfall, haze or NO₂.
2. **Pick a place**: Karachi, Lahore or Delhi.
3. **Pick the years** (2005 to 2024).
4. **See the time series.** The app asks what trend you see before it shows the statistics, so you act like a data detective first.
5. **Get the result:** the rate of change per decade, a 95% range, whether it is statistically significant, and a one-sentence verdict in plain English.

On top of that, it tests **possible reasons**:

- **Smog:** do Lahore and Delhi haze go together with crop-fire counts in October to December?
- **Karachi heat:** is the city warming faster than the land around it?

## The questions we are asking

| Question | What we look at |
|---|---|
| Why is smog worse in Lahore and Delhi? | Aerosol haze, NO₂, crop-fire counts |
| Why is Karachi getting hotter? | Air temperature, land surface temperature (city vs. outskirts) |
| Why does Karachi seem to get less rain? | Monthly rainfall from two independent sources |

"Less rain" is something we test, not something we assume. If a trend isn't there, the app says so. That is a valid result too.

## How the analysis works

All the maths lives in `analyze.py`. For every time series it does the following:

1. **Checks data quality.** It refuses very short or mostly missing records. It never fills gaps silently.
2. **Removes the seasons.** It compares each January with other Januaries, so the yearly cycle stops hiding the slow change.
3. **Measures the change.** It computes the Theil–Sen slope per decade with a 95% confidence range. This method isn't thrown off by outliers.
4. **Tests if it is real.** It runs the Mann–Kendall test. If the data is autocorrelated, it uses a corrected version, and it adds a seasonal Mann–Kendall check.
5. **Checks the shape.** It looks for a sudden jump (Pettitt test) and compares against a smoothed curve (LOWESS), and warns when a straight-line trend isn't a fair summary.
6. **Writes the verdict.** The sentence is built only from the computed numbers.

For possible reasons it uses Spearman correlation (also after removing each series' own trend, because two rising series can look linked by accident), with an optional lag. For Karachi, it runs the trend test on *city minus surroundings*.

**No machine learning.** On about 20 years of monthly data, classical trend tests are the right tool, and anyone can check them. The app also shows an "if this trend continues" band. It is clearly labelled as that, not as a prediction.

## Data

Monthly data, 2005 to 2024.

| Variable | Places | Source |
|---|---|---|
| Aerosol (haze) | Lahore, Delhi | MERRA-2 via NASA Giovanni |
| NO₂ | Lahore, Delhi | OMI via NASA Giovanni |
| Crop-fire counts | Punjab (India and Pakistan) | NASA FIRMS (MODIS) |
| Air temperature | Karachi, Lahore, Delhi | NASA POWER |
| Rainfall | Karachi | NASA POWER and GPM IMERG |
| Land surface temperature | Karachi city, Karachi outskirts | MODIS (via AppEEARS) |

Each file is a CSV with the columns `date,place,variable,value,unit`, one row per month. Missing values are left empty, never zero. The files are in `backend/data/nasa_data_clean/`.

## Run it locally

```bash
git clone <this repository>
cd <repository folder>
pip install streamlit plotly pandas numpy scipy statsmodels pymannkendall matplotlib
streamlit run app.py
```

Use the command line without the app, from the `backend/` folder:

```bash
# trend of one series
python analyze.py trend data/nasa_data_clean/temperature_karachi.csv --start 2005 --end 2024

# smog season only (October to December)
python analyze.py trend data/nasa_data_clean/aerosol_lahore.csv --start 2005 --end 2024 --months 10 11 12

# do two series go together?
python analyze.py compare data/nasa_data_clean/aerosol_lahore.csv data/nasa_data_clean/fires_punjab.csv --months 10 11 12 --max-lag 2

# is the city warming faster than its surroundings?
python analyze.py diff data/nasa_data_clean/landtemp_karachicity.csv data/nasa_data_clean/landtemp_karachirural.csv --start 2005 --end 2024
```

Run the tests with `python -m pytest -v`. They use synthetic data with a known trend, so we can check the tool finds what we planted, and that it rarely invents a trend in pure noise.

## What this tool can and can't tell you

We tried to be honest about the limits.

- **It shows what goes together, not what causes what.** A link between haze and fires is evidence, not proof.
- **Aerosol optical depth is not PM2.5.** It measures haze through the whole air column, not the air you breathe at ground level.
- **Twenty years is short.** A trend that isn't significant yet might become significant with a longer record.
- **Results depend on the chosen boxes.** Satellite grid cells are coarse, so a small area average can hide local effects.
- **Two sources can disagree.** When they do, we show both.

## The team

We are six people from biotechnology, computer science and full-stack development, some of us working full time. 
[Anas](linkedin.com/in/anasalpha/)
[Zoha](https://www.linkedin.com/in/syeda-zoha-fatima-7a0b013b2)
[Kalsoom]
[Ahmed Mujtaba](https://www.linkedin.com/in/ahmed-e-mujtaba-361a3a318?)
[Usama]

## Acknowledgements

Thanks to NASA and its data teams for open access to Earth science data: Giovanni, GES DISC, NASA POWER, MODIS, OMI, MERRA-2, GPM IMERG and FIRMS.

This project was built for the NASA Space Apps Challenge. It is not affiliated with or endorsed by NASA.
