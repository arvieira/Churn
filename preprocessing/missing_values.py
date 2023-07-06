# Procura por missing values
def missing_values(df):
    if df.isna().any().any() or df.isnull().any().any():
        print("\t-> Missing value encontrado. Realize o tratamento")
        return True
    else:
        print(f"\t-> Não foram encontrados Missing Values.")
        return False
