import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
 

# Step 1: Generate Random Data from a Predefined ARIMA Model
# Define AR (AutoRegressive) and MA (Moving Average) components
ar_params = np.array([0.6, -0.3])  # AR(2) process # Time series model: current value depends on two previous values and random noise
# X_t = 0.6 * X_{t-1} - 0.3 * X_{t-2} + ϵ_t
ma_params = np.array([0.4])  # MA(1) process
# X_t = μ + ϵ_t + 0.4 * ϵ_{t-1}  
#  
# Interpretacija:  
# Trenutna vrednost časovne vrste (X_t) je odvisna od:  
# 1. Povprečne vrednosti časovne vrste (μ),  
# 2. Trenutne naključne napake (ϵ_t),  
# 3. Pretekle naključne napake (ϵ_{t-1}), pomnožene z utežjo 0.4.  

# X_t = 0.6 * X_{t-1} - 0.3 * X_{t-2} + ϵ_t + 0.4 * ϵ_{t-1}
# Interpretacija:
# Model ARMA(2,1) - avtoregresivni drseči povprečje model 2. reda z drsečim povprečjem 1. reda

# Define ARMA process (ARIMA without differencing)
ar = np.r_[1, -ar_params]  # Include lag=0 coefficient np.r_: Uporablja se za združevanje vrednosti v eno polje (array).
ma = np.r_[1, ma_params]   # Include lag=0 coefficient
# prvi element v vektorjih koeficientov za AR in MA vedno 1 - To je standardna konvencija 
# X_t = 1.0 * X_t - 0.6 * X_{t-1} + 0.3 * X_{t-2} + ϵ_t

 
# Simulate 300 data points
np.random.seed(42)  # For reproducibility
simulated_data = ArmaProcess(ar, ma).generate_sample(nsample=300)
#Simulirani podatki predstavljajo časovno vrsto, ki temelji 
# na kombinaciji preteklih vrednosti in preteklih napak, kot je definirano z AR in MA komponentami.
 

# Convert to Pandas Series Pandas Series omogoča enostavno obdelavo,
# vizualizacijo in analizo podatkov, saj podpira funkcionalnosti, kot so indeksiranje po času.
time_series = pd.Series(simulated_data)

 

# Step 2: Check Self-Similarity (Autocorrelation)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(time_series, ax=axes[0], title="Autocorrelation (ACF) - for parameter q")
# Graf ACF prikazuje kako močno so trenutne vrednosti odvisne od preteklih vrednosti.
#Pomaga pri določanju parametra q (število zamikov premikajočega povprečja) za MA komponento ARIMA modela.
plot_pacf(time_series, ax=axes[1], title="Partial Autocorrelation (PACF) - for parameter p")
#Graf PACF prikazuje Neposredno povezavo med trenutnimi vrednostmi in preteklimi vrednostmi.
#Pomaga pri določanju parametra p (število zamikov autoregresije) za AR komponento ARIMA modela.
plt.show()

#Interpretacija grafa: Na prvem zamiku (lag = 1) je korelacija zelo visoka, kar pomeni, da trenutna vrednost močno korelira s prejšnjo vrednostjo.
#Korelacija na višjih zamikih postopoma upada in postane statistično nepomembna (stolpci znotraj modrih črt).
#To kaže na prisotnost komponente premikajočega povprečja (MA), saj so pomembni zamiki omejeni na začetne vrednosti.

#Interpretacija grafa: Na prvem zamiku (lag = 1) je korelacija zelo visoka, kar kaže na močan neposreden vpliv prve pretekle vrednosti na trenutno vrednost.
# Na drugem zamiku (lag = 2) je korelacija še vedno pomembna, vendar manjša.
# Korelacija na višjih zamikih hitro postane statistično nepomembna (stolpci znotraj modrih črt).
# To kaže na prisotnost komponente autoregresije (AR), saj so pomembni le začetni zamiki.

# Če je temperatura danes (X_t) neposredno odvisna od temperature včeraj (X_{t-1}),
# bo PACF za lag=1 visoka.

# Če je temperatura danes odvisna tudi od temperature predvčerajšnjim (X_{t-2}),
# vendar le prek včerajšnje temperature (X_{t-2} → X_{t-1} → X_t),
# bo PACF za lag=2 nizka, saj odstrani posredni vpliv.


# Perform the Augmented Dickey-Fuller test on the original series
from statsmodels.tsa.stattools import adfuller
result_original = adfuller(time_series)

print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")

if result_original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")


#additional step test train data Split data into train and test
train_size = int(len(time_series) * 0.8)
train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]


# Step 3: Fit an ARIMA Model to the Data (p=2, d=0, q=1)
model = ARIMA(train, order=(2, 0, 1))  # AR(2), I(0), MA(1) - to dobimo iz grafa kjer je izven meje
fitted_model = model.fit()

# Funkcija .fit() v ARIMA modelu oceni koeficiente ϕ₁, ϕ₂ (za AR komponento)
# in θ₁ (za MA komponento) na podlagi podatkov (time_series).
# Ti ocenjeni parametri niso nujno enaki začetnim parametrom (ar_params in ma_params)
# iz simulacije, ker se prilagodijo tako, da najbolje ustrezajo dejanskim podatkom.


# Step 4: Forecast Future Values

forecast_steps = len(test) #50  # Number of future steps
forecast = fitted_model.forecast(steps=forecast_steps)


# Step 5: Plot Original Data and Forecast
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Simulated Data", color='blue')
plt.plot(test.index, test, label="Test Data", color='green')
plt.plot(test.index, forecast, label="Forecast", color='red', linestyle="dashed")
plt.title("ARIMA Model Fit and Forecast")
plt.legend()
plt.show()


# rolling forecast
from sklearn.metrics import mean_squared_error

# Inicializacija
history = [x for x in train]  # Začetni učni podatki
predictions = []  # Seznam za shranjevanje napovedi

# Iterativno napovedovanje
for t in range(len(test)):
    # Prileganje modela na trenutne učne podatke
    model = ARIMA(history, order=(2, 0, 1))
    model_fit = model.fit()
    
    # Napoved ene prihodnje vrednosti
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    
    # Dodaj dejansko vrednost iz testnega sklopa v učne podatke
    history.append(test.iloc[t])

# Izračun napake (npr. RMSE)
rmse = mean_squared_error(test, predictions)
print(f"RMSE: {rmse}")

# Vizualizacija rezultatov
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Simulated Data", color='blue')
plt.plot(test.index, test, label="Test Data", color='green')
plt.plot(test.index, predictions, label="Rolling Forecast", color='red', linestyle="dashed")
plt.title("Rolling Forecast with ARIMA")
plt.legend()
plt.show()
