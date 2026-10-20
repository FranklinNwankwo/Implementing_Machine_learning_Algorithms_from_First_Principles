"""Streamlit Community Cloud entry point for the fraud-detection decision lab.

The app lives in a folder whose path contains spaces
("1. Supervised_Learning/02. logistic_regression_credit_card_fraud/app").
Streamlit Cloud passes the path of a requirements file found next to the main
module to its installer unquoted, so a dependency file there fails to install.
Deploying this root-level file instead makes Cloud use the root
requirements.txt, and this launcher runs the real app unchanged.

Local use is unaffected: `streamlit run app.py` inside the app folder still works.
"""

import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "1. Supervised_Learning" / "02. logistic_regression_credit_card_fraud" / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))  # so `import fraudlab` resolves

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
