# Breast Cancer Detector

A Streamlit dashboard that predicts whether a tumor is Benign (B) or Malignant (M) using a pre-trained scikit-learn model.

## Working dashboard

You can interact with the live, hosted dashboard here:

https://breast-cancer-detector-pjluyxibbx8swmevoe9fmz.streamlit.app/

This is a deployed, working instance of the project where you can try the model without running it locally.

## Repository contents

- `app.py` — Streamlit application (UI + prediction logic).
- `breast_cancer.csv` — (Optional) dataset used for exploration or training.
- `model.pkl` — Pre-trained scikit-learn model (required for local runs).
- `scaler.pkl` — Pre-processing scaler used before prediction (required for local runs).
- `requirements.txt` — Python dependencies.
- `notebook.ipynb` — Notebook used for exploration and model training.

## Quick start (Windows PowerShell)

1. Open PowerShell and change to the project directory:

```powershell
cd "D:\Linkedln Projects\Breast-Cancer-Detector"
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell execution policy, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser; .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r .\requirements.txt
```

4. Run the dashboard locally:

```powershell
streamlit run app.py
# or: python -m streamlit run app.py
```

The app will usually open at http://localhost:8501. If it does not, open that URL manually.

## Quick smoke test (verify model files load)

Run this to check the pickles load correctly:

```powershell
python -c "import pickle; m=pickle.load(open('model.pkl','rb')); s=pickle.load(open('scaler.pkl','rb')); print('Loaded model:', type(m), 'has predict:', hasattr(m,'predict'))"
```

## Troubleshooting

- FileNotFoundError for `model.pkl` / `scaler.pkl`: Ensure these files are present in the project root. If missing, either obtain them from the original source or re-train the model using `notebook.ipynb`.
- Pickle incompatibility: If loading fails with a protocol/version error, the pickles were likely created with a different Python or scikit-learn version. Try to match the training environment or re-train the model.
- Streamlit command not found: Make sure the virtual environment is active and `streamlit` is installed.

## Notes

- `app.py` expects the input features in a fixed order and uses `scaler.pkl` to scale inputs before prediction.
- The app will show a probability for the malignant class if the model supports `predict_proba`.

## License

This repository does not declare a license. Add a `LICENSE` file if you intend to publish.

---

If you want, I can add a small screenshot or badge linking to the working dashboard to make the link more visible.
