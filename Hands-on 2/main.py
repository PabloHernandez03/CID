import pandas as pd
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.teta1 = None
        self.teta0 = None
    
    def fit(self, X, y):
        x_2_suma = (X**2).sum()
        y_suma = y.sum()
        x_suma = X.sum()
        x_y = (X*y).sum()
        x_suma_2 = (x_suma)**2
        n = X.count()

        self.teta0 = ((x_2_suma*y_suma)-(x_suma*x_y))/((n*(x_2_suma))-x_suma_2)
        self.teta1 = ((n*x_y)-(x_suma*y_suma))/((n*(x_2_suma))-x_suma_2)

        print(f"Modelo ajustado: y = {self.teta1:.2f}x + {self.teta0:.2f}")
    
    def predict(self, X):
        return self.teta1 * X + self.teta0
    


def main():
    df = pd.read_csv("Hands-on 2/dataset.csv")

    print(df)
    X = df["Sales"]
    y = df["Advertising"]

    modelo = SimpleLinearRegression()

    modelo.fit(X, y)

    while True:
        entrada = input("Ingresa un valor para predecir (escriba 'fin' para finalizar): ")

        # Cambia 'valor' por 'entrada' en esta línea
        if entrada.lower() == 'fin':
            break

        try:
            valor = float(entrada)

            prediccion = modelo.predict(valor)

            print(f"Predicción para {valor} es {prediccion}")

        except ValueError:
            print("Error: Por favor, ingresa un número válido.")

    sales = pd.Series([0,2000])
    advertising = modelo.predict(sales)

    plt.scatter(X, y, color="blue", label='Data', marker='o')
    plt.plot(sales, advertising, color='red', label='Modelo', linewidth=2)

    plt.title('Caso Benetton')
    plt.xlabel('Advertising')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()



# Punto de entrada del programa
if __name__ == "__main__":
    main()