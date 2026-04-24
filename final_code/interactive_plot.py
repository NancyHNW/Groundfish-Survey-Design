import numpy as np
import matplotlib.pyplot as plt
from util.distance import *
from data_loader import *
from itertools import permutations, combinations, product
import math
import random
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import sys
from gui_funcs import *
from gui_global_vars import *
from classes import Trip


if __name__ == "__main__":
    # --- Root window ---
    root = tk.Tk()
    root.title("Port & Station Route Builder")
    root.geometry("1800x800")

    # --- Main layout frame ---
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # --- Left: Plot Area ---
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # --- Left Middle: Remaining/Visited Stations ---
    middle_frame = tk.Frame(main_frame)
    middle_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    # --- Right: Controls ---
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    # --- Matplotlib Figure and Canvas ---
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.1)
    draw_map(ax)
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()


    # --- Toolbar ---
    toolbar = NavigationToolbar2Tk(canvas, left_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    r_v_label = ttk.Label(middle_frame, text="Remaining/Visited Stations:")
    r_v_label.pack(pady=(10, 0))

    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Remaining/Visited Stations in Treeview ---
    remaining_visited_tree = ttk.Treeview(
        middle_frame,
        columns=(),
        show="tree",
        selectmode="browse",
        height=15
    )
    remaining_visited_tree.pack(padx=10, pady=14, fill=tk.BOTH, expand=True)

    remaining_visited_tree.insert("", "end", iid="remain", text="Remaining Stations", tags=("remain",))
    remaining_visited_tree.insert("", "end", iid="visited", text="Visited Stations", tags=("visited",))

    for i in range(len(station_ids)):
        remaining_visited_tree.insert("remain", "end", iid=f"{i+13}", text=f"{i+13}", tags=("station",))

    # --- Click event ---
    canvas.mpl_connect(
        "button_press_event",
        lambda event: on_click(event, ax, canvas, boat_trip_tree, remaining_visited_tree)
    )

    # --- Controls ---
    tt_label = ttk.Label(right_frame, text="Boat i (Capacity, Home Port):")
    tt_label.pack(pady=(10, 0))

    if not boat_capacities:
        ask_boat_info(root)

    tree_container = tk.Frame(right_frame)
    tree_container.pack(padx=10, pady=(14, 0), fill=tk.X)

    boat_trip_tree = ttk.Treeview(
        tree_container,
        columns=("main",),
        show="tree",
        selectmode="browse",
        height=15
    )
    boat_trip_tree.column("main", width=2000, stretch=False)
    boat_trip_tree.heading("main", text="Trips")
    boat_trip_tree.pack(side=tk.TOP, fill=tk.X)

    scroll_x = tk.Scrollbar(tree_container, orient="horizontal", command=boat_trip_tree.xview)
    boat_trip_tree.configure(xscrollcommand=scroll_x.set)
    scroll_x.pack(side=tk.BOTTOM, fill="x")

    for i in range(len(boat_capacities)):
        boat_objs.append(Boat(i+1, boat_capacities[i], home_ports[i]))
        boat_trip_tree.insert("", "end", iid=f"{i+1}", text=f"Boat {i+1} ({boat_capacities[i]}kg, {home_ports[i]})", tags=("boat",))
        boat_trip_tree.insert(f"{i+1}", "end", iid=f"trip:{i+1}:1", text=f"0.0h, 0.0h, 0.0kg", values= (f'trip:{i+1}:1: [{home_ports[i]}]', ), tags=("trip",))

    boat_trip_tree.bind(
        "<ButtonRelease-1>",
        lambda event: on_boat_trip_tree_select(event, ax, canvas, boat_trip_tree, total_time_label, total_fish_time_label, total_catch_label)
    )
    
    
    # Styling: bold boats vs. trips
    style = ttk.Style()
    style.configure("Treeview", rowheight=20)  # nicer spacing
    boat_trip_tree.tag_configure("boat", font=("TkDefaultFont", 10, "bold"))
    boat_trip_tree.tag_configure("trip", font=("TkDefaultFont", 10), foreground="#444")

    # --- listbox buttons ---
    button_row = tk.Frame(right_frame)
    button_row.pack()

    # undo_last_button = ttk.Button(button_row, text="Undo", command=lambda: undo_last())
    # undo_last_button.pack(side=tk.LEFT, padx=5)

    add_trip_small_button = ttk.Button(button_row, text="Add Trip", command=lambda: add_trip_small(root, ax, canvas, boat_trip_tree))
    add_trip_small_button.pack(side=tk.LEFT, padx=5)

    remove_trip_button = ttk.Button(button_row, text="Remove Trip", command= lambda: remove_trip(ax, canvas, boat_trip_tree, remaining_visited_tree))
    remove_trip_button.pack(side=tk.LEFT, padx=5)

    # --- other buttons ---
    below_buttons_frame = tk.Frame(right_frame)
    below_buttons_frame.pack(pady=15)

    total_time_label = ttk.Label(below_buttons_frame, text="Total Time (current trip): 0.0h")
    total_time_label.pack(pady=5)

    total_fish_time_label = ttk.Label(below_buttons_frame, text="Total Fish Time (current trip): 0.0h")
    total_fish_time_label.pack(pady=5)

    total_catch_label = ttk.Label(below_buttons_frame, text="Total Catch (current trip): 0.0kg")
    total_catch_label.pack(pady=5)

    ttk.Label(below_buttons_frame, text="Go to Node:").pack(pady=(10, 0))
    node_entry = ttk.Entry(below_buttons_frame)
    node_entry.pack(pady=5)
    node_entry.bind("<Return>", lambda event: 
                    go_to_node(node_entry, ax, canvas, boat_trip_tree))

    # go_button = ttk.Button(below_buttons_frame, text="Go to Node",
    #                        command=lambda: go_to_node(node_entry, ax, canvas, boat_trip_tree))
    # go_button.pack(pady=5)

    # test_button = ttk.Button(below_buttons_frame, text='test', command = lambda: test(root, ax, canvas, all_nice_trips_listbox))
    # test_button.pack(pady=5)

    go_to_nearest_port_button = ttk.Button(below_buttons_frame, text="Go to Nearest Port",
                                command=lambda: go_to_nearest_port(ax, canvas, boat_trip_tree, remaining_visited_tree))
    go_to_nearest_port_button.pack(pady=5)

    heuristic_button = ttk.Button(below_buttons_frame, text='Heuristic', command = lambda: heuristic_me(ax, canvas, root, boat_trip_tree, remaining_visited_tree))
    heuristic_button.pack(pady=5)

    initial_solution_button = ttk.Button(below_buttons_frame, text='Generate Initial Solution', 
                                         command = lambda: initial_solution_me(root, ax, canvas, boat_trip_tree, remaining_visited_tree))
    initial_solution_button.pack(pady=5)

    load_txt_button = ttk.Button(below_buttons_frame, text="Load from .txt",
                                    command=lambda: load_txt(root, ax, canvas, boat_trip_tree, remaining_visited_tree))
    load_txt_button.pack(pady=5)

    save_to_txt_button = ttk.Button(below_buttons_frame, text="Save Trips to .txt",
                                    command=lambda: save_to_txt())
    save_to_txt_button.pack(pady=5)


    previous_selection = [None]
    reset_button = ttk.Button(below_buttons_frame, text="Reset All",
                            command=lambda: reset(ax, canvas, boat_trip_tree, remaining_visited_tree))
    reset_button.pack(pady=10)

    for boat in boat_objs:
        Prob.boats.append(boat)


    # # Scrollbar
    # scrollbar = tk.Scrollbar(right_frame)
    # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # # Text widget (acts like a terminal)
    # output_text = tk.Text(right_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=20, width=50)
    # output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # scrollbar.config(command=output_text.yview)

    # # Redirect stdout and stderr
    # sys.stdout = StdoutRedirector(output_text)
    # sys.stderr = StdoutRedirector(output_text)

    # --- Exit handler ---
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(fig, root))

    # --- Start GUI --
    root.mainloop()
