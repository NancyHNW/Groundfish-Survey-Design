FILE OVERVIEW

This project contains three key Python files used to build and run an interactive Tkinter-based route visualisation tool for fishing boats. Here's what each file does and how they talk to each other.

------------------------------------------------------------
1. gui_global_vars.py 
Purpose:
Holds global data structures and constants shared across the GUI — especially objects that need to persist (like boats, trips, station IDs, etc).
Done so stuff that is used a lot like referencing the current Problem class or boat objs doesn't need to be an input into every function in gui_funcs

Key contents:
- boat_objs: list of Boat objects  
- ports_idx, station_ids: ID reference lists  
- nodes_converted_to_points: maps node IDs to matplotlib coordinates  
- Any values that must be accessed or modified across multiple modules  
- Problem class

------------------------------------------------------------
2. gui_funcs.py 
Purpose:
Contains logic functions used by the GUI: things like drawing trips, handling selections, resetting, saving, updating, etc.

Key functions:
- update_tree_big(...)
- weight_colour_indicator(...)
- on_click(...), on_trip_select(...)
- reset(...), save_to_txt(...), go_to_node(...)

Uses:
- Uses global_vars for shared state
- Called by GUI callbacks
- Can be used by heuristics if passed in as argument

------------------------------------------------------------
3. interactive_plot.py 
Purpose:
The main GUI runner – creates windows, packs frames, embeds Matplotlib, builds Treeviews, connects buttons.

Contains:
- tk.Frame(...), Treeview(...), Button(...)
- root.mainloop()
- Calls to functions in gui_funcs.py


------------------------------------------------------------
CIRCULAR IMPORTS

Avoid importing a file back into itself or importing two files into eachother.
(e.g. 1 -> 2 -> 3 -> 1 or 1 -> 2 -> 1) 

If there is a function you want to use - update_tree()- , dont import it, have it as an input into the desired function - next_descent(update = None) - and call it like this - update_callback=lambda: update_tree(...).

Inside next_descent:
def next_descent(..., update_callback):
    update_callback()
------------------------------------------------------------
GENERAL ADVICE

- Keep global_vars minimal
- draw_idle() is fast; draw() + root.update() shows live updates
- Use Treeview.item(...) to update/move items
- get_children() lets you iterate over IDs in the 
- Stack previous states for undo
