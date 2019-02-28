import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from numba import jit

def compute_scores(word_sets):
    word_nb = len(word_sets)
    score_array = np.zeros((word_nb, word_nb))
    for i in tqdm(range(word_nb)):
        set_i = word_sets[i]
        size_i = len(set_i)
        for j in range(i+1, word_nb):
            inter_nb = len(set_i.intersection(word_sets[j]))
            if (i == 2 and j==792):
                print(set_i, size_i, word_sets[j], len(word_sets[j]))
            score_array[i,j] = min(inter_nb, size_i - inter_nb, len(word_sets[j]) - inter_nb)
            score_array[j,i] = score_array[i,j]
    return score_array


def load_vertical(data_path, data_name):
    or_data_v = np.load("Save\\assignment_dct_100.npy").item()
    or_data_v = or_data_v[data_name]

    v_ids = or_data_v['assignment']
    v_tags = or_data_v['tags_save']
    v_tags = [set(tag_list) for tag_list in v_tags]

    return v_tags, v_ids

def load(data_path):
    # Load data
    or_data = np.load("Save\\data_dct.npy").item()
    data_name = 'a_examplet'

    # print(or_data.keys())
    # print(machin['a_examplet'].iloc[0, :])
    # Get ID Horizontal + IDs vertical pairs
    #for i in range()
    or_data = or_data[data_name] 
    or_data_h = or_data[or_data['H_V'] == 'H']
    id_h = list(or_data_h.iloc[:,0])

    # Get sets of tags
    word_strings = or_data_h.iloc[:,4]
    word_sets = [set([int(str_idx) for str_idx in s.split(" ")]) for s in word_strings]
    
    v_tags, v_ids = load_vertical(data_path, data_name)

    print(v_tags, v_ids)

    ids = id_h + v_ids
    word_sets = word_sets + v_tags

    print(word_sets)
    
    # Compute array
    score_array = compute_scores(word_sets)
    # Save the array
    np.save("tsp_score_array.npy", score_array)
    return score_array, ids

def greedy_solver(score_array):
    vertex_idx = 0
    chosen_vertices = np.zeros(score_array.shape[0], dtype=bool)
    chosen_vertices[vertex_idx] = True
    vertex_order = [vertex_idx]
    for idx in range(score_array.shape[0]-1):
        next_idx = np.argmax(score_array[vertex_idx, :] * np.invert(chosen_vertices) - 1000*chosen_vertices)
        #print(score_array[vertex_idx, :] * np.invert(chosen_vertices))
        vertex_order.append(next_idx)
        #print(next_idx)
        chosen_vertices[next_idx] = True
        vertex_idx = next_idx
    return np.array(vertex_order)


    # Virer le min dans le circuit

def TSP_solver(score_array):
    greedy_solution = greedy_solver(score_array)

    return greedy_solution

def make_output(greedy_solution, df_name, ids):
    list_sol = [ids[greedy_solution[idx]] for idx in greedy_solution]
    return {df_name: list_sol}

start_t = time()

score_array, ids = load('r')
print(score_array)
sol = TSP_solver(score_array)

dico_sol = make_output(sol, "b_lovely_landscapest", ids)
print(dico_sol)

end_t = time()
print(end_t - start_t)