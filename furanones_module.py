# модуль, подготовленный для практики - тут процесс подготовки данных, -
# стандартизация (в т ч распределение по файлам "ушедших" соединений),
# функция подсчета количества дубликатов на одну молекулу для стерео и 2д
# структур и функция распределения по классам для каждой мишени (в ней
# же запись классов каждой мишени и цитотоксичности в отдельный файл)
# то же самое представлено в файле work(и там еще процесс тестирования
# каких-то других уже неважных вещей)
#   в файле test.ipynb представлен следующий этап работы
from chython.files import SDFRead, SDFWrite
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter


def standartization_sdf(file_path):
    neorg_mol = 0
    flag_is = 0
    flag_coord = 0
    flag_salts = 0
    flag_neutr = 0
    flag_components = 0
    flag_radicals = 0
    flag_final_molecules = 0
    m_ishod = 0
    with SDFRead(file_path) as f, \
        SDFWrite(f'{file_path}_neorg_mol.sdf') as neorg_file, SDFWrite(f'{file_path}_сonnected_compounds.sdf') as compounds_file, \
        SDFWrite(f'{file_path}_Radicals_parts.sdf') as radical_file, SDFWrite(f'{file_path}_Clean_furanones_stereo.sdf') as clean_ster_furanones, \
        SDFWrite(f'{file_path}_Clean_furanones_2d.sdf') as clean_2d_furanones:  # записывает в sdf
        for i, m in enumerate(f):  # Он по-умолчанию дозаписывает? Да
            m_ishod += 1
            unique_atoms = set([a[1].atomic_symbol for a in list(m.atoms())])  # уникальные атомы, для проверки неорганики
            if 'C' not in unique_atoms:
                neorg_file.write(m)
                neorg_mol += 1
            else:
                can_log = m.canonicalize(logging=True)
                m.meta['canonicalized'] = can_log
                if m.clean_isotopes():  # убирает метки изотопов. Вернет True, если потребовалось
                    flag_is += 1
                if m.remove_coordinate_bonds():  # удаляет водор связи
                    flag_coord += 1
                if m.split_metal_salts(logging=True):  # разбивает металл соли
                    flag_salts += 1
                if m.neutralize():  # нейтрализует соли
                    flag_neutr += 1
                if m.connected_components_count > 1:  # после манипуляции с солями, есть шанс, что соед развалилось на 2,-> такие не нужны, отдел файл
                    flag_components += 1
                    compounds_file.write(m)
                else:
                    if m.is_radical:
                        flag_radicals += 1
                        radical_file.write(m)
                    else:
                        flag_final_molecules += 1
                        clean_ster_furanones.write(m)  # запись в подчищ файл со стереометками
                        m.clean_stereo()
                        clean_2d_furanones.write(m)  # запись в подчищ файл в формате 2d 
    # Создание DataFrame с результатами
    results = pd.DataFrame(
        data={"Критерий отбора":  [
            "Количество проанализированных молекул",
            "Количество отсеянных неорганических молекул",
            "Количество молекул, в которых были убраны изотопные метки",
            "Количество молекул, в которых были убраны водородные связи",
            "Количество разбитых металлических солей",
            "Количество нейтрализованных солей",
            "Количество отсеянных развалившихся надвое солей",
            "Количество отсеянных радикалов",
            "Количество конечных отобранных молекул"
        ],
        "Результат": [
            m_ishod,
            neorg_mol,
            flag_is,
            flag_coord,
            flag_salts,
            flag_neutr,
            flag_components,
            flag_radicals,
            flag_final_molecules
        ]
    })
    # Вывод результатов в консоль
    print(f'''Стандартизация SDF файла {file_path} с молекулами прошла успешно.
Все молекулы отсортированы по соответствующим файлам.
Результат стандартизации представлен ниже в таблице.
Статистика по стандартизации также отгружена в pickle файл.''')
    print(results.to_string(index=False, justify='center'))
    # Сохранение в pickle-файл
    results.to_pickle("statistics_of_standardization_results.pkl")
    
    
def count_duplicate_molecules(sdf_file_ster_path, sdf_file_2d_path):
    with SDFRead(sdf_file_ster_path) as f1, \
         SDFRead(sdf_file_2d_path) as f2:
        flag_m1_ish = 0
        flag_m2_ish = 0
        duplicates_list1 = defaultdict(list)  # {smiles: [id1, id2, ...]}
        duplicates_list2 = defaultdict(list)  # {smiles: [id1, id2, ...]}
        for m1 in f1:
            flag_m1_ish += 1
            id1 = m1.meta['id']
            smiles1 = str(m1)  # Генерируем SMILES
            duplicates_list1[smiles1].append(id1)
        # Подсчитываем количество вхождений каждого SMILES
        smiles_counts1 = Counter({smi1: len(ids1) for smi1, ids1 in duplicates_list1.items()})
        # Группируем молекулы по количеству дубликатов
        duplicates_count1 = Counter(smiles_counts1.values())
        for m2 in f2:
            flag_m2_ish += 1
            id2 = m2.meta['id']
            smiles2 = str(m2)  # Генерируем SMILES
            duplicates_list2[smiles2].append(id2)
        # Подсчитываем количество вхождений каждого SMILES
        smiles_counts2 = Counter({smi2: len(ids2) for smi2, ids2 in duplicates_list2.items()})
        # Группируем молекулы по количеству дубликатов
        duplicates_count2 = Counter(smiles_counts2.values())
    # Выводим 1 статистику
    print(f'Анализ SDF файла {sdf_file_ster_path} до перевода в 2D: ')
    for dup_count1 in duplicates_count1.keys():
        if dup_count1 == 1:
            print(f'Всего проанализировано молекул: {flag_m1_ish}')
        else:
            print(f"Молекул с {dup_count1-1} дубликатами: {duplicates_count1[dup_count1]}")
    # Выводим 2 статистику
    print(f'\nАнализ SDF файла {sdf_file_2d_path} после перевода в 2D: ')
    for dup_count2 in duplicates_count2.keys():
        if dup_count2 == 1:
            print(f'Всего проанализировано молекул: {flag_m2_ish}')
        else:
            print(f"Молекул с {dup_count2-1} дубликатами: {duplicates_count2[dup_count2]}")


def class_filtrating(file_path):
    with SDFRead(file_path) as mols:
            general_list = defaultdict(list)  # это класс, создаем словарь словарей списков
            mic_list = ['МПК Aspergillus niger, мкг/мл','МПК Penicillium funiculosum, мкг/мл','МПК Fusarium solani, мкг/мл','МПК Mucor pusillus, мкг/мл','МПК Alternaria alternata, мкг/мл']
            for m in mols:
                id = m.meta['id']
                ic50 = m.meta['IC50']
                if ic50 == "–":
                    ic50 = None
                elif ic50 == ">128":
                    ic50 = None
                else:
                    ic50 = float(ic50)
                general_list['IC50'].append([id, ic50])
                for target in mic_list:  # перебор
                    mic = m.meta[target]  # присваивание конкретного значения mic
                    try:
                        mic_val = float(mic)
                    except ValueError:
                        category = 6
                    else:
                        category = int(np.digitize(mic_val, [8, 16, 32, 64, 128], right=True))  # присваивание класса каждому объекту мишени
                    general_list[target].append([id, category])  # попадет в соответствующий словарь и в каждом словаре у нас будет список списков
            for target, target_list in general_list.items():  # создание DataFrame для каждой мишени и запись в pickle файл
                if target == 'IC50':
                    df = pd.DataFrame.from_records(target_list, columns=['id','IC50'])
                else:
                    df = pd.DataFrame.from_records(target_list, columns=['id','class'])
                df.to_pickle(f'{target.replace(', мкг/мл','').replace(' ','_')}.pkl')
    print('Проведено распеделение по классам каждой мишени. Успешно созданы следующие файлы:')
    for target in general_list.keys():
        print(f'{target.replace(', мкг/мл','').replace(' ','_')}.pkl')