def plot_acf_pacf(timeseries):
    fig = plt.figure(figsize=(12, 18))
    for index, (timeserie_title, timeserie) in enumerate(timeseries.items()):
        index = index * 2
        ax = fig.add_subplot(len(timeseries), 2, index + 1)
        ax.title.set_text(timeserie_title)
        graphics.plot_acf(timeserie, ax=ax, lags=40)
        ax.title.set_text('ACF %s' % timeserie_title)

        ax = fig.add_subplot(len(timeseries), 2, index + 2)
        graphics.plot_pacf(timeserie, ax=ax, lags=40)
        ax.title.set_text('PACF %s' % timeserie_title)
        

log_catfish_sales = numpy.log(catfish_sales)

# création d'un graphique avec la série originale
g, ax = plt.subplots()
ln1 = ax.plot(catfish_sales, c='r', label='Ventes de poissons-chats (1000s de livres)')

# création d'un graphique avec la série log en conservent le même axe des abscisses
ax2 = ax.twinx()
ln2 = ax2.plot(log_catfish_sales, c='b', label='Log(Ventes de poissons-chats)')

# ajout de la légende
lns = ln1 + ln2
labels=[l.get_label() for l in lns]
ax.legend(lns, labels)

# Séparer les données en ensemble d'entraînement et ensemble de test
train_data = catfish_sales[:-15]
test_data = catfish_sales[-15:]

# Utiliser auto_arima pour trouver le meilleur modèle ARIMA
# model = pm.auto_arima(train_data)
model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
 

print(model.summary())

# Ajuster le modèle aux données
model.fit(train_data)
# Obtenir les résidus du modèle
residuals = model.resid()
plot_acf_pacf({
    'Residue modèle': residuals
})

# Faire des prédictions sur l'ensemble d'entraînement
train_pred, train_confint = model.predict_in_sample(return_conf_int=True)

# Faire des prédictions sur l'ensemble de test
n_periods = len(test_data)
predicted, confint = model.predict(n_periods=n_periods, return_conf_int=True)

# Concaténer les prédictions pour l'ensemble d'entraînement et de test
all_predictions = pandas.concat([pandas.Series(train_pred, index=train_data.index),
                            pandas.Series(predicted, index=test_data.index)],
                            axis=0)

# Tracer les valeurs réelles et les prédictions pour l'ensemble d'entraînement et de test
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Ventes observées (Entraînement)', color='blue')
plt.plot(test_data, label='Ventes observées (Test)', color='green')
plt.plot(all_predictions, label='Ventes prédites', color='red')

plt.xlabel('Date')
plt.ylabel('Ventes de poissons-chats (1000s de livres)')
plt.title('Ventes observées vs prédites de poissons-chats')
plt.legend()
plt.grid(True)
# plt.savefig("prediction.png")
plt.show()
