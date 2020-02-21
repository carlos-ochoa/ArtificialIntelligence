import itertools
import random
from operator import itemgetter
import matplotlib.pyplot as plt

# Cities dictionary
cities = \
    {'Ciudad de México':
            {'Ecatepec':24,
            'Guadalajara':535,
            'Puebla':130,
            'Tijuana':2827,
            'Juárez':1805,
            'León':375,
            'Zapopan':541,
            'Monterrey':899,
            'Neza':16,
            'Hermosillo':1888,
            'Chihuahua':1440,
            'Naucalpan':12,
            'Mérida':1306,
            'San Luis Potosí':402,
            'Aguascalientes':490,
            'Saltillo':836,
            'Mexicali':2640,
            'Culiacán':1215,
            'Acapulco':381}
        ,
    'Ecatepec':
            {'Ciudad de México':24,
            'Guadalajara':532,
            'Puebla':142,
            'Tijuana':2816,
            'Juárez':1794,
            'León':364,
            'Zapopan':538,
            'Monterrey':888,
            'Neza':27,
            'Hermosillo':1885,
            'Chihuahua':1429,
            'Naucalpan':32,
            'Mérida':1319,
            'San Luis Potosí':391,
            'Aguascalientes':479,
            'Saltillo':825,
            'Mexicali':2629,
            'Culiacán':1212,
            'Acapulco':404}
    ,
    'Guadalajara':
            {'Ciudad de México':535,
            'Ecatepec':532,
            'Puebla':668,
            'Tijuana':2229,
            'Juárez':1538,
            'León':224,
            'Zapopan':7,
            'Monterrey':795,
            'Neza':547,
            'Hermosillo':1361,
            'Chihuahua':1173,
            'Naucalpan':533,
            'Mérida':1844,
            'San Luis Potosí':331,
            'Aguascalientes':223,
            'Saltillo':705,
            'Mexicali':2058,
            'Culiacán':688,
            'Acapulco':866}
    ,
    'Puebla':
            {'Ciudad de México':130,
            'Ecatepec':142,
            'Guadalajara':668,
            'Tijuana':2957,
            'Juárez':1935,
            'León':505,
            'Zapopan':671,
            'Monterrey':1029,
            'Neza':123,
            'Hermosillo':2018,
            'Chihuahua':1570,
            'Naucalpan':143,
            'Mérida':1184,
            'San Luis Potosí':532,
            'Aguascalientes':620,
            'Saltillo':966,
            'Mexicali':2770,
            'Culiacán':1345,
            'Acapulco':426}
    ,
    'Tijuana':
            {'Ciudad de México':2827,
            'Ecatepec':2816,
            'Guadalajara':2229,
            'Puebla':2957,
            'Juárez':1183,
            'León':2463,
            'Zapopan':2223,
            'Monterrey':2195,
            'Neza':2840,
            'Hermosillo':873,
            'Chihuahua':1388,
            'Naucalpan':2824,
            'Mérida':4130,
            'San Luis Potosí':2406,
            'Aguascalientes':2334,
            'Saltillo':2114,
            'Mexicali':217,
            'Culiacán':1569,
            'Acapulco':3216}
    ,
    'Juárez':
            {'Ciudad de México':1805,
            'Ecatepec':1794,
            'Guadalajara':1538,
            'Puebla':1935,
            'Tijuana':1183,
            'León':1444,
            'Zapopan':1546,
            'Monterrey':1176,
            'Neza':1821,
            'Hermosillo':838,
            'Chihuahua':368,
            'Naucalpan':1805,
            'Mérida':3111,
            'San Luis Potosí':1387,
            'Aguascalientes':1314,
            'Saltillo':1094,
            'Mexicali':999,
            'Culiacán':1103,
            'Acapulco':2196}
    ,
    'León':
            {'Ciudad de México':375,
            'Ecatepec':1040,
            'Guadalajara':224,
            'Puebla':505,
            'Tijuana':2463,
            'Juárez':1444,
            'Zapopan':228,
            'Monterrey':692,
            'Neza':388,
            'Hermosillo':1575,
            'Chihuahua':1074,
            'Naucalpan':372,
            'Mérida':1678,
            'San Luis Potosí':179,
            'Aguascalientes':125,
            'Saltillo':607,
            'Mexicali':2274,
            'Culiacán':902,
            'Acapulco':763}
    ,
    'Zapopan':
            {'Ciudad de México':541,
            'Ecatepec':538,
            'Guadalajara':7,
            'Puebla':671,
            'Tijuana':2223,
            'Juárez':1546,
            'León':228,
            'Monterrey':801,
            'Neza':553,
            'Hermosillo':1353,
            'Chihuahua':1111,
            'Naucalpan':539,
            'Mérida':1850,
            'San Luis Potosí':337,
            'Aguascalientes':229,
            'Saltillo':711,
            'Mexicali':2051,
            'Culiacán':680,
            'Acapulco':871}
    ,
    'Monterrey':
            {'Ciudad de México':899,
            'Ecatepec':888,
            'Guadalajara':795,
            'Puebla':1029,
            'Tijuana':2195,
            'Juárez':1176,
            'León':692,
            'Zapopan':801,
            'Neza':913,
            'Hermosillo':1493,
            'Chihuahua':805,
            'Naucalpan':897,
            'Mérida':2203,
            'San Luis Potosí':514,
            'Aguascalientes':573,
            'Saltillo':88,
            'Mexicali':2005,
            'Culiacán':1039,
            'Acapulco':1288}
    ,
    'Neza':
            {'Ciudad de México':16,
            'Ecatepec':27,
            'Guadalajara':547,
            'Puebla':123,
            'Tijuana':2840,
            'Juárez':1821,
            'León':388,
            'Zapopan':553,
            'Monterrey':913,
            'Hermosillo':1901,
            'Chihuahua':1453,
            'Naucalpan':26,
            'Mérida':1298,
            'San Luis Potosí':415,
            'Aguascalientes':503,
            'Saltillo':849,
            'Mexicali':2653,
            'Culiacán':1228,
            'Acapulco':384}
    ,
    'Hermosillo':
            {'Ciudad de México':1888,
            'Ecatepec':1885,
            'Guadalajara':1361,
            'Puebla':2018,
            'Tijuana':873,
            'Juárez':838,
            'León':1575,
            'Zapopan':1353,
            'Monterrey':1493,
            'Neza':1901,
            'Chihuahua':691,
            'Naucalpan':1882,
            'Mérida':3193,
            'San Luis Potosí':1602,
            'Aguascalientes':1530,
            'Saltillo':1416,
            'Mexicali':699,
            'Culiacán':696,
            'Acapulco':2215}
    ,
    'Chihuahua':
            {'Ciudad de México':1440,
            'Ecatepec':1429,
            'Guadalajara':1173,
            'Puebla':1570,
            'Tijuana':1388,
            'Juárez':368,
            'León':1074,
            'Zapopan':1111,
            'Monterrey':805,
            'Neza':1453,
            'Hermosillo':691,
            'Naucalpan':1439,
            'Mérida':2745,
            'San Luis Potosí':1021,
            'Aguascalientes':949,
            'Saltillo':729,
            'Mexicali':1202,
            'Culiacán':738,
            'Acapulco':1831}
    ,
    'Naucalpan':
            {'Ciudad de México':12,
            'Ecatepec':32,
            'Guadalajara':533,
            'Puebla':143,
            'Tijuana':2824,
            'Juárez':1805,
            'León':372,
            'Zapopan':539,
            'Monterrey':897,
            'Neza':26,
            'Hermosillo':1882,
            'Chihuahua':1439,
            'Mérida':1317,
            'San Luis Potosí':396,
            'Aguascalientes':484,
            'Saltillo':829,
            'Mexicali':2634,
            'Culiacán':1213,
            'Acapulco':395}
    ,
    'Mérida':
            {'Ciudad de México':1306,
            'Ecatepec':1319,
            'Guadalajara':1844,
            'Puebla':1184,
            'Tijuana':4130,
            'Juárez':3111,
            'León':1678,
            'Zapopan':1850,
            'Monterrey':2203,
            'Neza':1298,
            'Hermosillo':3193,
            'Chihuahua':2745,
            'Naucalpan':1317,
            'San Luis Potosí':1713,
            'Aguascalientes':1802,
            'Saltillo':2147,
            'Mexicali':3951,
            'Culiacán':2526,
            'Acapulco':1620}
    ,
    'San Luis Potosí':
            {'Ciudad de México':402,
            'Ecatepec':391,
            'Guadalajara':331,
            'Puebla':532,
            'Tijuana':2406,
            'Juárez':1387,
            'León':179,
            'Zapopan':337,
            'Monterrey':514,
            'Neza':415,
            'Hermosillo':1602,
            'Chihuahua':1021,
            'Naucalpan':396,
            'Mérida':1713,
            'Aguascalientes':164,
            'Saltillo':448,
            'Mexicali':2217,
            'Culiacán':933,
            'Acapulco':790}
    ,
    'Aguascalientes':
            {'Ciudad de México':490,
            'Ecatepec':479,
            'Guadalajara':223,
            'Puebla':620,
            'Tijuana':2334,
            'Juárez':1314,
            'León':125,
            'Zapopan':229,
            'Monterrey':573,
            'Neza':503,
            'Hermosillo':1530,
            'Chihuahua':949,
            'Naucalpan':484,
            'Mérida':1802,
            'San Luis Potosí':164,
            'Saltillo':479,
            'Mexicali':2147,
            'Culiacán':863,
            'Acapulco':878}
    ,
    'Saltillo':
            {'Ciudad de México':836,
            'Ecatepec':825,
            'Guadalajara':705,
            'Puebla':966,
            'Tijuana':2114,
            'Juárez':1094,
            'León':607,
            'Zapopan':711,
            'Monterrey':88,
            'Neza':849,
            'Hermosillo':1416,
            'Chihuahua':729,
            'Naucalpan':829,
            'Mérida':2147,
            'San Luis Potosí':448,
            'Aguascalientes':479,
            'Mexicali':1924,
            'Culiacán':959,
            'Acapulco':1223}
    ,
    'Mexicali':
            {'Ciudad de México':2640,
            'Ecatepec':2629,
            'Guadalajara':2058,
            'Puebla':2770,
            'Tijuana':2114,
            'Juárez':999,
            'León':2274,
            'Zapopan':2051,
            'Monterrey':2005,
            'Neza':2653,
            'Hermosillo':699,
            'Chihuahua':1202,
            'Naucalpan':2634,
            'Mérida':3951,
            'San Luis Potosí':2217,
            'Aguascalientes':2147,
            'Saltillo':1924,
            'Culiacán':1394,
            'Acapulco':3031}
    ,
    'Culiacán':
            {'Ciudad de México':1215,
            'Ecatepec':1212,
            'Guadalajara':688,
            'Puebla':1345,
            'Tijuana':217,
            'Juárez':1103,
            'León':902,
            'Zapopan':680,
            'Monterrey':1039,
            'Neza':1228,
            'Hermosillo':696,
            'Chihuahua':738,
            'Naucalpan':1213,
            'Mérida':2526,
            'San Luis Potosí':933,
            'Aguascalientes':863,
            'Saltillo':959,
            'Mexicali':1394,
            'Acapulco':1545}
    ,
    'Acapulco':
            {'Ciudad de México':381,
            'Ecatepec':404,
            'Guadalajara':866,
            'Puebla':426,
            'Tijuana':3216,
            'Juárez':2196,
            'León':763,
            'Zapopan':871,
            'Monterrey':1288,
            'Neza':384,
            'Hermosillo':2215,
            'Chihuahua':1831,
            'Naucalpan':395,
            'Mérida':1620,
            'San Luis Potosí':790,
            'Aguascalientes':878,
            'Saltillo':1223,
            'Mexicali':3031,
            'Culiacán':1545}
    }

# PARAMETERS OF THE ALGORITHM
USE_ELITISM = False
ELITISM = 0
CROSSOVER = 0.8
MUTATION = 0.2
N = 500
GENERATIONS = 100
ROUNDS = 3

# Class Solution
class Solution():
    genotype = []
    fitness = 0
    selected_for_elitism = False

    def __init__(self, genotype):
        self.genotype = genotype
        self.calculate_fitness()

    def calculate_fitness(self):
        distance = 0
        distance += cities['Ciudad de México'][self.genotype[0]]
        for i in range(18):
            distance += cities[self.genotype[i]][self.genotype[i+1]]
        self.fitness = distance

    def get_fitness(self):
        return self.fitness

    def set_for_elitism(self,val):
        self.selected_for_elitism = val

    def get_genotype(self):
        return self.genotype


def calculate_fitness(population):
    fitness = {}
    distance = 0
    for key,ind in population.items():
        distance += cities['Ciudad de México'][ind[0]]
        for i in range(18):
            distance += cities[ind[i]][ind[i+1]]
        fitness[key] = distance
        distance = 0
    return fitness

# Return list of indexes
def tournament_selection(population, expected_breeds):
    selected_population = []
    champ = None
    champ_index = None
    parents_needed = int(((N - ELITISM) * CROSSOVER)) + int((N - ELITISM) * MUTATION)
    # Organizing a tournament with ROUNDS
    for i in range(parents_needed):
        champ = None
        for r in range(ROUNDS):
            index = random.randint(0,N-1)
            fighter2 = population[index]
            if champ == None:
                champ = fighter2
            elif fighter2.get_fitness() < champ.get_fitness():
                champ = fighter2
        selected_population.append(champ)
    return selected_population

def find_unused(source,used):
    for x in source:
        if not x in used:
            used.append(x)
            return [x,used]

def slice_crossover_n(parents,cut_length):
    offspring1 = [0 for i in parents[0].get_genotype()]
    offspring2 = [0 for i in parents[0].get_genotype()]
    used1, used2 = [] , []
    cutpoint1 = int(random.uniform(0,len(parents[0].get_genotype())- 1 - cut_length))
    cutpoint2 = cutpoint1 + cut_length
    print(cutpoint1)
    print(cutpoint2)
    # Middle part
    for i in range(len(parents[0].get_genotype()) - 1):
        if i >= cutpoint1 and i < cutpoint2:
            offspring1[i] = parents[1].get_genotype()[i]
            offspring2[i] = parents[0].get_genotype()[i]
            used1.append(offspring1[i])
            used2.append(offspring2[i])
    # Outer part
    for i in range(len(parents[0].get_genotype())):
        if i < cutpoint1 or i >= cutpoint2:
            res = find_unused(parents[0].get_genotype(), used1)
            offspring1[i] = res[0]
            used1 = res[1]
            res = find_unused(parents[1].get_genotype(), used2)
            offspring2[i] = res[0]
            used2 = res[1]
    print(offspring1)
    print(offspring2)
    return [Solution(offspring1),Solution(offspring2)]



def make_couples(parents):
    current_couple = []
    couples = []
    for ind in parents:
        current_couple.append(ind)
        if len(current_couple) == 2:
            couples.append(current_couple.copy())
            current_couple.clear()
    return couples

def slice_crossover_nr(parents,cuts):
    cut_point = random.randint(0,len(parents[0].get_genotype()))
    offspring1 = parents[0].get_genotype()[:cut_point]
    offspring2 = parents[1].get_genotype()[:cut_point]
    used = offspring1.copy()
    used2 = offspring2.copy()
    # Offsrping 1
    j = cut_point
    for i in parents[1].get_genotype()[cut_point:]:
        if i not in used:
            offspring1.append(i)
            used.append(i)
        else:
            offspring1.append(parents[0].get_genotype()[j])
            used.append(parents[0].get_genotype()[j])
        j += 1
    # Offsrping 2
    j = cut_point
    for i in parents[0].get_genotype()[cut_point:]:
        if i not in used2:
            offspring2.append(i)
            used2.append(i)
        else:
            offspring2.append(parents[1].get_genotype()[j])
            used2.append(parents[1].get_genotype()[j])
        j += 1
    print(cut_point)
    print(str(parents[0].get_genotype()) + '\n')
    print(str(parents[1].get_genotype()) + '\n')
    print(str(offspring1) + '\n')
    print(str(offspring2) + '\n')
    return [Solution(offspring1),Solution(offspring2)]

def shuffle_mutation(individuals):
    print(len(individuals))
    mutated = []
    for ind in individuals:
        index1 = random.randint(0,len(ind.get_genotype())-1)
        index2 = random.randint(0,len(ind.get_genotype())-1)
        aux = ind.get_genotype()[index1]
        ind.get_genotype()[index1] = ind.get_genotype()[index2]
        ind.get_genotype()[index2] = aux
        p = Solution(ind.get_genotype())
        mutated.append(p)
    return mutated

# Generate the first population randomly
population = []
items = list(cities.keys())
items.pop(0)
for i in range(N):
    random.shuffle(items)
    population.append(Solution(items.copy()))

best = []
gens = []

for gen in range(GENERATIONS):
    expected_breeds = N
    crossed = []
    mutated = []
    selected_population = []
    print(len(population))
    if USE_ELITISM:
        selected_population += sorted(population , key = lambda ind : ind.fitness, reverse = False )[:ELITISM]
        expected_breeds -= ELITISM
    # Selection step
    winners = tournament_selection(population, expected_breeds)
    print('Selection done!')
    # Crossover operation
    # We need only CROSSOVER % of the winners list
    cross_pop = int(((N - ELITISM) * CROSSOVER))
    parents = make_couples(winners[:cross_pop])
    for c in parents:
        crossed += slice_crossover_n(c,5)
    selected_population += crossed
    print('Crossover done!')
    # Mutation Operation
    mutated = shuffle_mutation(winners[cross_pop:])
    print('Mutation done!')
    selected_population += mutated
    population = selected_population.copy()
    better_solution = sorted(population , key = lambda ind : ind.fitness, reverse = False )[0]
    best.append(better_solution.get_fitness())
    gens.append(gen)
    print('Better solution: ' + str(better_solution.get_genotype()) + ' at generation ' + str(gen+1) + ' with fitness ' + str(better_solution.get_fitness()))

plt.plot(gens,best)
plt.show()
