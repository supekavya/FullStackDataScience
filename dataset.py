import pandas as pd
import numpy as np

def create_dataset():
    np.random.seed(42)

    data = {
        "Gender": np.random.choice(["Male", "Female"], 500),
        "Married": np.random.choice(["Yes", "No"], 500),
        "Education": np.random.choice(["Graduate", "Not Graduate"], 500),
        "ApplicantIncome": np.random.randint(2000, 20000, 500),
        "LoanAmount": np.random.randint(50, 500, 500),
        "Credit_History": np.random.choice([0,1], 500, p=[0.2,0.8]),
        "Property_Area": np.random.choice(["Urban","Rural","Semiurban"], 500),
    }

    df = pd.DataFrame(data)

    # Simple rule to generate target
    df["Loan_Status"] = np.where(
        (df["Credit_History"] == 1) &
        (df["ApplicantIncome"] > 4000),
        "Approved", "Rejected"
    )

    df.to_csv("loan_data.csv", index=False)
    return df

if __name__ == "__main__":
    create_dataset()
    print("Dataset Created!")
