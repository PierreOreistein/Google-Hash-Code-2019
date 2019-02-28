import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def compute_scores(word_sets):
    word_nb = len(word_sets)
    score_array = np.zeros((word_nb, word_nb))
    for i in range(word_nb):
        set_i = word_sets[i]
        size_i = len(set_i)
        for j in range(i+1, word_nb):
            inter_nb = len(set_i.intersection(word_sets[j]))
            if (i == 2 and j==792):
                print(set_i, size_i, word_sets[j], len(word_sets[j]))
            score_array[i,j] = min(inter_nb, size_i - inter_nb, len(word_sets[j]) - inter_nb)
            score_array[j,i] = score_array[i,j]
    return score_array

def load(data_path):
    # Load data
    or_data = np.load("Save\\data_dct.npy").item()
    # print(or_data.keys())
    # print(machin['a_examplet'].iloc[0, :])
    # Get ID Horizontal + IDs vertical pairs
    #for i in range()
    or_data = or_data['b_lovely_landscapest']
    or_data_h = or_data[or_data['H_V'] == 'H']
    id_h = list(or_data_h.iloc[:,0])[:1000]
    
    # Get sets of tags
    word_strings = or_data_h.iloc[:,4]
    word_sets = [set([int(str_idx) for str_idx in s.split(" ")]) for s in word_strings][:1000]
    # Compute array
    score_array = compute_scores(word_sets)
    # Save the array
    np.save("tsp_score_array.npy", score_array)
    return score_array, id_h

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
print(score_array[2])
sol = TSP_solver(score_array)

dico_sol = make_output(sol, "b_lovely_landscapest", ids)
#print(dico_sol)

end_t = time()
print(end_t - start_t)