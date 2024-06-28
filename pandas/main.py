import pandas as pd
import warnings

# Establecer la configuración de advertencias en "ignore"
warnings.filterwarnings("ignore")

# Leemos el excel con los datos
data = pd.read_csv('countries.csv')
data = data.rename(columns= {'gdpPercap': 'PIB'})
# 1. ¿A qué continente pertenece Tunisia?
continente_Tunisia = data[data.country == "Tunisia"].continent.iat[0]
print(f'Tunisia esta en el continente: {continente_Tunisia}')

# 2. ¿En que países la esperanza de vida fue mayor a 80 en el 2007?
paisMazLifeExp = list(data[data.lifeExp > 80][data.year == 2007].country)
print(f'los paises con esperanza de vida mayor a 80 en el 2007: {paisMazLifeExp}')

# 3. ¿Que país de América tiene el mayor PIB?
paisMaxPIB = data[(data.country == 'Paraguay') | (data.country == 'Venezuela')][data.year == 1967].sort_values('pop', ascending=False).country.iat[0]
print(f'El pais con mayor PIB en america es: {paisMaxPIB}')

# 4. ¿Qué país tenia más habitantes en 1967 entre Venezuela y Paraguay?
masHabitantes = data[(data.country == 'Paraguay') | (data.country == 'Venezuela')][data.year == 1967].sort_values('pop', ascending=False).country.iat[0]
print(f'El pais con mas habitantes en 1967 entre venzuela y paraguay: {masHabitantes}')

# 5. ¿En que año Panamá alcanzó una esperanza de vida mayor a 60 años?
lifeExp60 = data[data.country == 'Panama'][data.lifeExp > 60].year.iat[0]
print(f'año Panamá alcanzó una esperanza de vida mayor a 60 años: {lifeExp60}')

# 6. ¿Cuál es el promedio de la esperanza de vida en África en 2007?
prom = data[data.continent == 'Africa'][data.year == 2007].lifeExp.mean()
print(f'El Promedio de la esperanza de vida en africa es : {prom}')

# 7. Enlista los países en que el PIB de 2007 fue menor que su PIB en 2002
Pib2002 = data[data.year == 2002].PIB.mean()
res = list(data[data.PIB < Pib2002][data.year == 2007].country)
print(f'países en que el PIB de 2007 fue menor que su PIB en 2002 : {res}')

# 8. ¿Qué país tiene más habitantes en 2007?
res = data[data.year == 2007].sort_values('pop', ascending=False).country.iat[0]
print(f'Qué país tiene más habitantes en 2007: {res}')

# 9. ¿Cuantos habitantes tiene América en 2007?
res = data[data.year == 2007].sort_values('pop', ascending=False).country.iat[0]
print(f'Cuantos habitantes tiene América en 2007: {res}')

# 10. ¿Qué continente tiene menos habitantes en 2007?
res = data[data.year == 2007].sort_values('pop', ascending=True).country.iat[0]
print(f'Qué continente tiene menos habitantes en 2007: {res}')

# 11. ¿Cuál es el promedio de PIB en Europa?
res = data[data.continent == 'Europe'].PIB.mean()
print(f'Cuál es el promedio de PIB en Europa: {res}')

# 12. ¿Cuál fue el primer país de Europa en superar los 70 millones de habitantes
res = data[(data['continent'] == 'Europe') & (data['pop'] >= 70000000)].sort_values('year').country.iat[0]
print(f'Cuál fue el primer país de Europa en superar los 70 millones de habitantes: {res}')