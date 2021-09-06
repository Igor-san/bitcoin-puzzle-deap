from deap import base
from deap import creator
from deap import tools

import random, time

from bit import Key

# адрес как цель поиска, для теста уже решенные
#TARGET_ADDRESS ="1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV" # №9 467  111010011 очень легкая для поиска
# длина бинарной последовательности нам известна
#target_len=9

# адрес как цель поиска
TARGET_ADDRESS ="1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE" # №18 198669  3080D  110000100000001101 для RANDOM_SEED = 8 потребовалось 209 популяций
target_len=18

#TARGET_ADDRESS ="16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN" # №64 длина 64 , первая еще не решенная!
#target_len=64

# константы задачи:
ONE_MAX_LENGTH = target_len  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма:
POPULATION_SIZE = 500 # количество индивидуумов в популяции
P_CROSSOVER = 0.8  # вероятность скрещивания
P_MUTATION = 0.2   # вероятность мутации индивидуума
MAX_GENERATIONS = 400 # максимальное количество поколений
TOUR_N_SIZE=4 # размер турнира для отбора
INDPB =0.1 # независимая вероятность инвертирования каждого бита mutFlipBit

# ГСЧ: В реальном приложении назначить None!
RANDOM_SEED = 8 # для воспроизводимости результатов нужно задать константу
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# регистрируем оператор возвращающий случайные 0 или 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# определяем единственную цель максимизируя фитнес-функцию:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# создаем класс Индивидуума основанном на списке:
creator.create("Individual", list, fitness=creator.FitnessMax)

# создаем индивидуальный оператор создающий экземпляр индивидуума
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# создаем популяционный оператор для создания списка индивидуумов
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Генетические операторы:

# Оператор турнирного отбора заданного размера:
toolbox.register("select", tools.selTournament, tournsize=TOUR_N_SIZE)

# Двухточетное скрещивания:
toolbox.register("mate", tools.cxTwoPoint)

# мутация переворотом:
# indpb: независимая вероятность инвертирования каждого бита
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
#toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB/ONE_MAX_LENGTH)

# расчет финтес-функции (ака приспособленность):
def oneMaxFitness(individual):
    number =int("".join(str(x) for x in individual), 2) # переводим из двоичного в десятичный
    
    if (number<=0): # для ключа только 1 и выше
        return (1,) # нужно вернуть кортеж
    guess = Key.from_int(number).address
    # попарно сравниваем символы в адресах и суммируем совпадения 
    summa = sum(1 for expected, actual in zip(TARGET_ADDRESS, guess)
               if expected == actual)

    return (summa,) # нужно вернуть кортеж

# регистрация функции приспособленности
toolbox.register("evaluate", oneMaxFitness)

# получить приспообленность, адрес, секрет из индивидуума для отображения
def getAddress(individual):
    secret="".join(str(x) for x in individual)
    number =int(secret, 2)
    if (number<=0):
        return (0,"","") # нужно вернуть кортеж

    guess = Key.from_int(number).address
    summa = sum(1 for expected, actual in zip(TARGET_ADDRESS, guess)
               if expected == actual)

    return (summa,guess,secret) # нужно вернуть кортеж


# Основная программа генетического алгоритма:
def run():

    # в Генерации 0 создаем начальную популяцию:
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # расчитываем фитнес-кортеж для каждого индивидуума в популяции
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # извлекаем значения приспособленности для всех индивидуумов:
    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitness=max(fitnessValues) # начальные значения на случай что угадаем сразу
    best_index = fitnessValues.index(max(fitnessValues))
    best_result=(maxFitness , population[best_index])

    # инициализируем массивы для статистики:
    maxFitnessValues = []
    meanFitnessValues = []


    # Главный эволюционный цикл:
    # условия остановки: если максимальная приспобленность достигнута
    # Или достигли предела числа поколений:
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1

        # применяем оператор выбора для генерации следующего поколения 
        offspring = toolbox.select(population, len(population))
        # клонируем выбранных особей:
        offspring = list(map(toolbox.clone, offspring))

        # применяем оператор скрещивания (с заданной вероятностью) к парам потомков
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values # удаляем приспособленность особи как флаг для перерасчета
                del child2.fitness.values # удаляем приспособленность особи как флаг для перерасчета

        # применяем оператор мутирования (с заданной вероятностью) к каждому потомку
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values # удаляем приспособленность мутанта как флаг для перерасчета

        # расчитываем вновь приспособленность индивидуумов с удаленными выше фитнесами
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # заменяем текущую популяцию потомками:
        population[:] = offspring

        # собираем новые финтесы в список и обновляем статистику
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)

        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)

        # находим лучшего индивидуума:
        best_index = fitnessValues.index(max(fitnessValues))
                
        if (maxFitness>best_result[0]):
            best_result=(maxFitness , population[best_index])

        pass

    # вышли из цикла, печатаем результат:
    result =False
    summa, address, secret=getAddress(best_result[1])
    print("Лучший индивидуум = ", best_result, "->",summa, " ->",address )
    if (address==TARGET_ADDRESS):
        result =True
        print("Загадка решена! Secret = ", secret, " потребовалось популяций = ", generationCounter)

    return result

if __name__ == '__main__':
    start = time.time()
    run()
    end = time.time()
    print(end - start)
