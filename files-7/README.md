# Market Validation Dashboard — Single-File Version

Everything is in **one file**. There are no imports between your own files, so
there is no module that can fail to upload.

## What your repo needs

```
app.py                          <- the entire application
requirements.txt
data/
  survey_responses.csv
  multiselect_onehot.csv
  multiselect_baskets_long.csv
  segment_answer_key.csv
  chart_register.csv
  survey_module2_strategy.csv
```

That's it. Two entries at the top level.

## Deploy to Streamlit Cloud

**If you already have a broken deployment, delete the old folder from GitHub first.**
The previous error persisted because the old `app.py` was still in the repo.

1. On GitHub, open your repo → delete the whole `files-5` folder
   (open the folder → each file → trash icon → commit). Or from the command line:
   ```bash
   git rm -r files-5
   git commit -m "remove old app"
   ```

2. Upload this folder's contents to the repo root:
   ```bash
   git add -A
   git commit -m "single-file app"
   git push
   ```

3. In Streamlit Cloud → **Manage app** → **Reboot app**. Set the main file path
   to `app.py`.

## Verify before deploying

Open `app.py` on GitHub and search for `lib.common`. If you find it, you are
still looking at the old file and the deploy will fail again. The correct file
contains no `lib` references at all and is about 1,900 lines.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages

Navigate with the sidebar radio buttons:

| Page | Technique | Result |
|---|---|---|
| Overview | Descriptive KPIs | Adoption, WTP, waste, barriers |
| Data Quality | Cleaning audit, MNAR diagnosis | 7 duplicates, 25% MNAR missingness |
| Segmentation | k-means | k=4, ARI 0.56 |
| Adoption Model | Logistic / random forest | ROC-AUC ~0.91, live lead scoring |
| Pricing Model | OLS regression | R² ~0.49 on willingness to pay |
| Association Rules | Apriori | 24 rules at lift > 1.25 |
| Dashboard Roadmap | — | 107 specified charts |
| Feature & GTM Strategy | Van Westendorp, feature scoring | Price point ₹13,000, roadmap decisions |

## Data caveat

The survey data is **synthetic** — generated from an assumed model of the Indian
food service market to demonstrate analytical method. Every relationship the app
recovers was designed into the generator. Valid as a demonstration of technique;
not evidence about the real market. State this in any presentation.
