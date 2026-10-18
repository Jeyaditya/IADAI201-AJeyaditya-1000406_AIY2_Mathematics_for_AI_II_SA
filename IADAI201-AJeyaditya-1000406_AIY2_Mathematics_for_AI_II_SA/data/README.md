# data/

Put the raw dataset here as **`data/raw_sparcs.csv`** (a `.csv.gz` also works). The delimiter (comma, tab, pipe)
and encoding are detected automatically.

| File | What it is | Commit it? |
| --- | --- | --- |
| `raw_sparcs.csv` | The original hospital discharge file from the course Drive link. | Yes (if your Git setup allows files this large) |
| `inpatient_clean.csv.gz` | Cleaned + feature-engineered **cache**, created automatically on first run. | **Yes** - makes the deployed app start fast and is the safety net if the raw file can't be read on the server |
| `cleaning_report.json` | Row counts for every cleaning step (shown in the app's *Data & cleaning* tab). | Yes |
| `clean_meta.json` | Remembers which raw file / row cap produced the cache. | Yes |

## How it works
1. The first time the app starts it cleans `raw_sparcs.csv` in chunks (a few minutes for a very large file) and
   keeps a uniform random sample of about **500,000** valid rows (`MAX_ROWS` in `app.py`).
2. The result is saved as `inpatient_clean.csv.gz`; later starts load that in seconds.
3. If you replace the raw file or change `MAX_ROWS`, the cache is rebuilt automatically.

Want every row (local use)? `MAX_ROWS=0 streamlit run app.py` (Windows PowerShell: `$env:MAX_ROWS=0; streamlit run app.py`).

Optional - watch progress in the terminal / pre-build the cache:
```bash
python prepare_data.py                     # uses data/raw_sparcs.csv
python prepare_data.py --max-rows 300000   # smaller sample
python prepare_data.py --max-rows 0        # all rows
```
