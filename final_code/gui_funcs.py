import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from util.distance import degArr2point
from gui_global_vars import *
from classes import Problem, Boat, Trip
from data_loader import *
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from better_tsp import solve_tsp
from neighbourhood_rules import *
import math as math

def find_nearest_station_or_port(click_coord):
    '''should work'''
    closest = [np.inf, None, False]  # [distance, node_id, is_port]

    for pid in ports_idx:
        location_on_plot = nodes_converted_to_points[pid]
        distance = np.linalg.norm(click_coord - location_on_plot)
        if distance < closest[0]:
            closest = [distance, pid, True]

    for sid in station_ids:
        # a, b are the two sides of the station
        a = sid * 2 + n_ports
        b = a + 1
        pa, pb = nodes_converted_to_points[a], nodes_converted_to_points[b]
        da = np.linalg.norm(click_coord - pa)
        db = np.linalg.norm(click_coord - pb)

        if da < closest[0]:
            closest = [da, sid + 13, False]
        if db < closest[0]:
            closest = [db, sid + 13, False]

    return [int(closest[1]), closest[2]] if closest[0] < 7 else None

def on_click(event, ax, canvas, boat_trip_tree, remaining_visited_tree, entered_node = None,):
    '''should work'''
    # print(boat_objs[1].route)
    if event is None:
        pass
    elif not event.inaxes:
        return

    if entered_node:
        station_port = entered_node
        if entered_node <= 12:
            is_port = True
        else:
            is_port = False
    else:
        click_xy = np.array([event.xdata, event.ydata])
        output = find_nearest_station_or_port(click_xy)

        if output == None:
            return
    
        station_port, is_port = output[0], output[1]

    # for trip in trip_objs:
    #     if not is_port and station_port in trip.nice_trip:
    #         print(f"station {station_port} is already in a trip")
    #         return
        
    for boat in boat_objs:
        for trip in boat.route:
            if not is_port and station_port in trip.nice_trip:
                print(f"station {station_port} is already in a trip")
                return

    selected = boat_trip_tree.selection()
    if len(selected) > 1:
        return
    
    make_new_trip = False
    
    # len(selected) is 0 or 1
    # if nothing selected, work on the last trip in the last boat in theboat_trip_tree
    if len(selected) == 0:
        last_index = len(boat_trip_tree.get_children("")) - 1
        current_boat = boat_objs[last_index]
        current_trip = current_boat.route[-1]
    elif selected[0].startswith("trip"):
        boat_num, trip_num = map(int, selected[0].split(":")[1:])
        current_boat = boat_objs[boat_num - 1]
        current_trip = current_boat.route[trip_num - 1]
    else: # selected is a boat
        boat_num = int(selected[0])
        current_boat = boat_objs[boat_num - 1]
        current_trip = current_boat.route[-1]
        # print(current_boat.route)
    # if (len(current_trip.nice_trip) >= 3 and current_trip.nice_trip[0] <= 12 and current_trip.nice_trip[-1] <= 12):
    #     if len(current_boat.route[-1].nice_trip) >= 3 and current_boat.route[-1].nice_trip[-1] <= 12:
    #         make_new_trip = True
    #     else:
    #         current_trip = current_boat.route[-1]
    # else: # current trip is not yet complete
    #     pass

    if make_new_trip:
        trip_id = f'trip:{current_boat.id}:{len(current_boat.route)+1}'
        current_trip = Trip(id = trip_id)
        current_boat.route.append(current_trip)
        current_trip = current_boat.route[-1]

    # if current trip is long and first node is a port and last node is a port, return
    if len(current_trip.nice_trip) >= 3 and current_trip.nice_trip[-1] <= 12: # and current_trip.nice_trip[0] <= 12:
        return
        
    # if current trip is empty and clicked node is not a port, return
    # as first node must be a port
    if current_trip.nice_trip == [] and not is_port:
        # print(f"first port error: {current_trip} \ntrip objs: {trip_objs}")
        print("First node must be a port")
        return


    if current_trip.nice_trip == [] and station_port != current_boat.route[-2].nice_trip[-1]:

        # print(f"first port error: {current_trip} \ntrip objs: {trip_objs}")
        print("First node must be the port the previous trip ended at")
        return
    
    # if fresh trip and clicked node is a port, append it to the trip to start the trip
    # if current_trip.nice_trip == [] and is_port:
    #     print("Starting new trip at port", station_port)
    #     boat_trip_tree.insert(f'{current_boat.id}', "end", iid=current_trip.id, text=f'{current_trip.id}: []', tags=("trip",))
    #     current_trip.nice_trip.append(station_port)
    #     update_tree_big(ax, canvas,boat_trip_tree, remaining_visited_tree)
    #     return
    
    # if last node in current trip is a port and clicked node is a port, return
    # as two ports in a row is not allowed
    if len(current_trip.nice_trip) == 1 and is_port:
        print("cannot add two ports in a row")
        return  

    # if current trip is real (real meaning it has started at a port and left that port) 
    # and clicked node is a port, append it to the trip (other logic handles ending current trip and starting new trip)
    if len(current_trip.nice_trip) >= 2 and station_port in ports_idx:
        current_trip.nice_trip.append(station_port)
        update_tree_add_port(ax, canvas, boat_trip_tree, remaining_visited_tree, station_port, current_trip)
        current_boat.route.append(Trip(start_port=station_port, id = f'trip:{current_boat.id}:{len(current_boat.route)+1}'))
        current_trip = current_boat.route[-1]
        boat_trip_tree.insert(f'{current_boat.id}', "end", iid=current_trip.id, text=f"0.0h, 0.0h, 0.0kg", values=(f'{current_trip.id}: [{current_trip.nice_trip[0]}]', ), tags=("trip",))
        return
    
    # add a satation to the current trip
    if station_port in station_ids:
        Prob.stations_left.remove(station_port)
        # print(f"nice trip before adding station: {current_trip.nice_trip}")
        # print(f"Clicked on: {station_port}")
        current_trip.nice_trip.append(station_port)
        # print(f"nice trip after adding station: {current_trip.nice_trip}")
        # print(current_boat.route)
        dict_station_num_annotations[station_port].set_alpha(1)
        update_tree_add_station(ax, canvas, boat_trip_tree, remaining_visited_tree, station_port, current_trip)
        # print(f"nice trip after listbox: {current_trip.nice_trip}")
        return

def ask_boat_info(root):
    """
    Popup dialog to ask number of boats, then collect capacities and home ports.
    Returns a list of dicts like:
      [{'capacity': 1000, 'home': 'Port A'}, {'capacity': 1200, 'home': 'Port B'}]
    """
    result = []

    # Create a modal dialog
    popup = tk.Toplevel(root)
    popup.title("Boat Setup")
    popup.geometry("400x400")
    popup.grab_set()  # make it modal

    tk.Label(popup, text="How many boats?").pack(pady=5)
    num_var = tk.IntVar(value=1)
    num_combo = ttk.Combobox(popup, textvariable=num_var, values=[1,2,3,4,5,6,7,8,9,10], state="readonly")
    num_combo.pack(pady=5)

    # Frame for dynamic rows
    rows_frame = tk.Frame(popup)
    rows_frame.pack(pady=10, fill="x")

    entries = []  # [(capacity_entry, home_entry), ...]

    def build_rows(*_):
        # clear old rows
        for w in rows_frame.winfo_children():
            w.destroy()
        entries.clear()

        for i in range(num_var.get()):
            row = tk.Frame(rows_frame)
            row.pack(fill="x", pady=2)

            tk.Label(row, text=f"Boat {i+1} capacity (kg):").pack(side="left", padx=5)
            cap_entry = ttk.Entry(row, width=8)
            cap_entry.pack(side="left")

            tk.Label(row, text="Home port (0-12):").pack(side="left", padx=5)
            home_entry = ttk.Entry(row, width=12)
            home_entry.pack(side="left")

            entries.append((cap_entry, home_entry))

    num_combo.bind("<<ComboboxSelected>>", build_rows)
    build_rows()  # initial build

    def submit():
        boats = []
        for i, (capacity_entry, home_entry) in enumerate(entries, start=1):
            try:
                capacity = float(capacity_entry.get())
                if capacity <= 0:
                    raise ValueError("Capacity must be positive")
            except ValueError:
                return  # stop here, don't close popup

            home = home_entry.get().strip()
            if not home:
                messagebox.showerror(
                    "Invalid input",
                    f"Please enter a home port for Boat {i}."
                )
                return

            boat_capacities.append(capacity)
            home_ports.append(home)

        # If we got this far, everything is valid → accept and close
        popup.destroy()

    ttk.Button(popup, text="Submit", command=submit).pack(pady=10)

    popup.wait_window()  # block until closed
    return result

def draw_map(ax):
    '''should work'''
    ax.plot(*island_data_converted_to_points, color = 'black', linewidth = 1)
    for i, idx in enumerate(ports_idx):
        port_point = degArr2point(nodes[[idx]])
        if i == 0:
            ax.scatter(*port_point, label='Port', s = 40)
        else: 
            ax.scatter(*port_point, s = 40)
        dict_station_num_annotations[idx] = ax.annotate(idx, (port_point[0][0], port_point[1][0]))
    for j, sid in enumerate(station_ids):
        station = nodes[[sid * 2 + n_ports, sid * 2 + 1 + n_ports], :]
        station_coord = degArr2point(station)
        if j == 0:
            ax.plot(*station_coord, "#459BEB", label='Station')
        else:
            ax.plot(*station_coord, "#459BEB")
        midpoint = [station_coord[0].mean(), station_coord[1].mean()]
        dict_station_num_annotations[sid+13] = ax.annotate(sid + 13, midpoint, fontsize = 9)
    ax.invert_yaxis()
    ax.legend(fontsize=10)

def reset(ax, canvas, boat_trip_tree, remaining_visited_tree):
    '''should work'''
    confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset?")
    if confirm:

        for remaining_visited in remaining_visited_tree.get_children(""):
            remaining_visited_tree.delete(*remaining_visited_tree.get_children(remaining_visited))

        for i in range(len(station_ids)):
            remaining_visited_tree.insert("remain", "end", iid=f"{i+13}", text=f"{i+13}", tags=("station",))

        for boat_iid in boat_trip_tree.get_children(""):
            boat_trip_tree.delete(*boat_trip_tree.get_children(boat_iid))
        Prob.reset(gui = True)
        for boat in boat_objs:
            boat_trip_tree.insert(f"{boat.id}", "end", iid=f"trip:{boat.id}:1", text=f'trip:{boat.id}:1: [{boat.home_port}]', tags=("trip",))
            
            for trip in boat.route:
                if len(trip.actual_trip) == 0 and len(trip.nice_trip) == 0:
                    print(trip.stations)
                    continue
                else:
                    for line in trip.route_lines:
                        line.remove()
                    trip.route_lines.clear()
            boat.route.clear()
            boat.route.append(Trip(start_port=boat.home_port, id = f'trip:{boat.id}:1'))
        canvas.draw_idle()

    else:
        return

def update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree, heuristic = False):
    '''should work'''
    
    for remaining_visited in remaining_visited_tree.get_children(""):
        remaining_visited_tree.delete(*remaining_visited_tree.get_children(remaining_visited))

    for i in range(len(station_ids)):
        remaining_visited_tree.insert("remain", "end", iid=f"{i+13}", text=f"{i+13}", tags=("station",))

    for boat_iid in boat_trip_tree.get_children(""):
        boat_trip_tree.delete(*boat_trip_tree.get_children(boat_iid))

    for boat in boat_objs:
        for trip in boat.route:
            # print(boat.route)
            trip.evaluate_total_time()
            trip.evaluate_fish_time()
            trip.evaluate_total_catch()
            if len(trip.actual_trip) == 0 and len(trip.nice_trip) == 0:
                continue
            else:
                for line in trip.route_lines:
                    line.remove()
                trip.route_lines.clear()
                # if nice and actual trips are not sync, get actual trip from nice trip
                if (not (len(trip.nice_trip) - 1)*2 == (len(trip.actual_trip) - 1) and \
                    not (len(trip.nice_trip) - 2)*2 == (len(trip.actual_trip) - 2)):
                    # print(len(trip.nice_trip), len(trip.actual_trip))
                    trip.get_actual_trip_from_nice_trip()
                if heuristic:
                    trip.actual_trip = trip.get_full_trip()
                    trip.get_nice_trip_from_actual_trip()
                x_coords = [nodes_converted_to_points[i][0] for i in trip.actual_trip.copy()]
                y_coords = [nodes_converted_to_points[i][1] for i in trip.actual_trip.copy()]
                weight_colour_indicator(ax, x_coords, y_coords, trip)
                # trip.route_lines.extend(ax.plot(x_coords, y_coords, color = 'black'))
                canvas.draw_idle()
                boat_trip_tree.insert(f"{boat.id}", "end", iid=trip.id, text = f"{trip.total_time:.1f}h, {trip.fish_time:.1f}h, {trip.total_catch:.1f}kg",values=(f"{trip.id}: {trip.nice_trip}",), tags=("trip",))
                for station in trip.nice_trip:
                    if station not in ports_idx:
                        remaining_visited_tree.move(station, "visited", "end")
        root.update_idletasks()
    sort_children(remaining_visited_tree, "remain")
    sort_children(remaining_visited_tree, "visited")

    # for idx, boat in enumerate(boat_trip_tree.get_children("")):
    #     print(boat_trip_tree.get_children(boat))
    #     print(boat_objs[idx].route)

# add a trip to the treeview, only change the boat that is affected
def update_tree_add_station(ax, canvas, boat_trip_tree, remaining_visited_tree, new_station_port, trip):
    # eval stuff and display it
    trip.evaluate_total_time()
    trip.evaluate_fish_time()
    trip.evaluate_total_catch()
    boat_trip_tree.item(f"{trip.id}", text = f"t{trip.total_time:.1f}h, f{trip.fish_time:.1f}h, {trip.total_catch:.1f}kg", values=(f"{trip.id}: {trip.nice_trip}",))
    
    # move station from remaining to visited
    remaining_visited_tree.move(new_station_port, "visited", "end")

    # remove lines from plot
    for line in trip.route_lines:
        line.remove()
    trip.route_lines.clear()

    # draw new line
    trip.get_actual_trip_from_nice_trip()
    x_coords = [nodes_converted_to_points[i][0] for i in trip.actual_trip.copy()]
    y_coords = [nodes_converted_to_points[i][1] for i in trip.actual_trip.copy()]
    weight_colour_indicator(ax, x_coords, y_coords, trip)
    canvas.draw_idle()

    sort_children(remaining_visited_tree, "remain")
    sort_children(remaining_visited_tree, "visited")

def update_tree_add_port(ax, canvas, boat_trip_tree, remaining_visited_tree, new_station_port, trip):
    # eval stuff and display it
    trip.evaluate_total_time()
    trip.evaluate_fish_time()
    boat_trip_tree.item(f"{trip.id}", text = f"{trip.total_time:.1f}h, {trip.fish_time:.1f}h, {trip.total_catch:.1f}kg", values=(f"{trip.id}: {trip.nice_trip}",))

    # remove lines from plot
    for line in trip.route_lines:
        line.remove()
    trip.route_lines.clear()

    # draw new line
    trip.get_actual_trip_from_nice_trip()
    x_coords = [nodes_converted_to_points[i][0] for i in trip.actual_trip.copy()]
    y_coords = [nodes_converted_to_points[i][1] for i in trip.actual_trip.copy()]
    weight_colour_indicator(ax, x_coords, y_coords, trip)
    canvas.draw_idle()

def weight_colour_indicator(ax, x_coords, y_coords, trip):
    print(trip.nice_trip)
    print(trip.id)
    boat_num, trip_num = map(int, trip.id.split(":")[1:])
    capacity = boat_objs[boat_num - 1].capacity

    catch_weights = []
    for i in range(1, len(trip.actual_trip)):
        if trip.actual_trip[i] <= 12:
            catch_weights.append(0)
        elif i % 2 == 1:
            catch_weights.append(float(trip.catch[int(math.floor((trip.actual_trip[i] - 13) / 2))]))
        elif i % 2 == 0:
            catch_weights.append(0)
    cumsum_weights = np.cumsum(np.array(catch_weights, dtype=float))
    # print(f"cumsum_weights: {cumsum_weights}")

    # fullness should be length = number of drawn segments; adjust as needed for your rule.
    fullness = cumsum_weights / capacity

    # build segments
    points = np.column_stack((x_coords, y_coords))
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)

    # --- Colormap/Normalize: create once per-axes and reuse ---
    if not hasattr(ax, "_fullness_norm"):
        ax._fullness_empty_hex = "#3EA1D3"
        ax._fullness_full_hex  = "#FF160D"
        ax._fullness_cmap = LinearSegmentedColormap.from_list(
            "boatfill", [ax._fullness_empty_hex, ax._fullness_full_hex]
        ).with_extremes(over="black")
        ax._fullness_norm = Normalize(vmin=0, vmax=1, clip=False)

    lc = LineCollection(segments, cmap=ax._fullness_cmap, norm=ax._fullness_norm, linewidths=2)
    lc.set_array(fullness)

    # keep a reference so you can remove later
    h = ax.add_collection(lc)
    trip.route_lines.append(lc)

    ax.autoscale()

    # --- Colorbar: create once as an inset, then update on later calls ---
    if not hasattr(ax, "_fullness_cbar"):
        # Small horizontal bar in the lower-left of the axes
        # width/height are relative to the axes size; tweak to taste
        ax._fullness_cax = inset_axes(
            ax, width="30%", height="3%",
            loc="lower left",  # position anchor
            bbox_to_anchor=(0.05, 0.05, 1, 1),  # x, y offsets (axes-fraction)
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        ax._fullness_cbar = plt.colorbar(h, cax=ax._fullness_cax, orientation="horizontal")
        ax._fullness_cbar.set_label("Boat Fullness", fontsize=8)
        ax._fullness_cbar.ax.tick_params(labelsize=8, length=2)
    else:
        # Reuse existing colorbar; just point it at the new mappable
        ax._fullness_cbar.update_normal(h)

def go_to_node(node_entry, ax, canvas, boat_trip_tree):
    '''AI'''
    try:
        node_id = int(node_entry.get())
    except ValueError:
        return
    on_click(None, ax, canvas, boat_trip_tree, node_id)
    node_entry.delete(0, tk.END)


def go_to_nearest_port(ax, canvas, boat_trip_tree, remaining_visited_tree):
    closest = [np.inf, None]  # [distance, node_id]

    selected = boat_trip_tree.selection()
    if len(selected) > 1:
        return
    
    # len(selected) is 0 or 1
    # if nothing selected, work on the last trip in the last boat in the boat_trip_tree
    if len(selected) == 0:
        last_index = len(boat_trip_tree.get_children()) - 1
        current_boat = boat_objs[last_index]
        current_trip = current_boat.route[-1]
    elif selected[0].startswith("trip"):
        boat_num, trip_num = map(int, selected[0].split(":")[1:])
        current_boat = boat_objs[boat_num - 1]
        current_trip = current_boat.route[trip_num - 1]
    elif selected[0].startswith("boat"):
        boat_num = int(selected[0])
        current_boat = boat_objs[boat_num - 1]
        current_trip = current_boat.route[-1]

    if current_trip.nice_trip[-1] <= 12:
        return
    
    for pid in ports_idx:
        location_on_plot = nodes_converted_to_points[pid]
        distance = np.linalg.norm(nodes_converted_to_points[current_trip.actual_trip[-1]] - location_on_plot)
        if distance < closest[0]:
            closest = [distance, pid]
    
    current_trip.actual_trip.append(int(closest[1]))
    current_trip.actual_trip = solve_tsp(current_trip.actual_trip, current_trip.time)
    current_trip.get_nice_trip_from_actual_trip()
    update_tree_big(ax, canvas,boat_trip_tree, remaining_visited_tree)

def load_txt(root, ax, canvas, boat_trip_tree, remaining_visited_tree):
    popup = tk.Toplevel(root)

    sol_label = tk.Label(popup, text="Enter File Name:")
    sol_label.pack(pady=5)

    sol_entry = tk.Entry(popup)
    sol_entry.pack(pady=5)


    def submit():
        sol_input = sol_entry.get().strip()

        # Use whichever one is filled
        if sol_input:
            with open(f'{sol_input}', 'r') as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines]

            stations = eval(lines[0])
            ports = eval(lines[1])
            fish_time_limit = eval(lines[2])
            n_boats = eval(lines[3])
            boat_capacities = eval(lines[4])
            home_ports = eval(lines[5])
            sol_list = eval(lines[6])

            Prob = Problem(list(stations), ports, fish_time_limit, n_boats, boat_capacities, home_ports)
            boat_objs.clear()
            for boat in Prob.boats:
                boat_objs.append(boat)
            Prob.restore_solution_from_list(sol_list)


            update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree, heuristic = False)

        popup.destroy()

    submit_button = tk.Button(popup, text="Submit", command=submit)
    submit_button.pack(pady=10)


def save_to_txt():
    Prob.write_problem_to_txt('saved_problem.txt')

def undo_last(clicked, ax, canvas):
    pass

def remove_trip(ax, canvas, boat_trip_tree, remaining_visited_tree):
    '''AI but works i think'''
    selected = boat_trip_tree.selection()
    if not selected:
        return
    for selection in selected:
        if selection.startswith("boat"):
            return
        else: # is a trip
            boat_num, trip_num = map(int, selected[0].split(":")[1:])
            current_boat = boat_objs[boat_num - 1]
            current_trip = current_boat.route[trip_num - 1]
            current_trip.route_lines[0].remove()
            for station in current_trip.nice_trip:
                if station not in ports_idx:
                    try:
                        remaining_visited_tree.move(station, "remain", "end")
                    except tk.TclError:
                        pass
            current_boat.route.remove(current_trip)
            if current_boat.route == []:
                current_boat.route.append(Trip(id = f'trip:{current_boat.id}:1'))
            update_tree_big(ax, canvas, boat_trip_tree, remaining_visited_tree)
            canvas.draw_idle()
    return

def on_boat_trip_tree_select(event, ax, canvas, boat_trip_tree, total_time_label, total_fish_time_label, total_catch_label):
    selected = boat_trip_tree.selection()
    
    # If no selection, restore all to normal
    if selected == previous_selection[0]:
        previous_selection[0] = None
        for boat in boat_objs:
            for trip in boat.route:
                for line in getattr(trip, "route_lines", []):
                    line.set_alpha(1.0)
        for annotation in dict_station_num_annotations.values():
            annotation.set_alpha(1.0)
        canvas.draw_idle()
        return

    previous_selection[0] = selected

    stations_in_trips = []
    if selected[0].startswith("trip"):
        for boat in boat_objs:
            for idx, trip in enumerate(boat.route, start=1):
                trip_iid = f"trip:{boat.id}:{idx}"
                if trip_iid == selected[0]:
                    for station in trip.nice_trip:
                        stations_in_trips.append(station)
                    # Highlight selected trip: fully opaque
                    for line in getattr(trip, "route_lines", []):
                        line.set_alpha(1.0)
                    trip.evaluate_total_time()
                    trip.evaluate_fish_time()
                    trip.evaluate_total_catch()
                    total_time_label.config(text=f"Total Time (current trip): {trip.total_time:.2f}h")
                    total_fish_time_label.config(text=f"Total Fish Time (current trip): {trip.fish_time:.2f}h")
                    total_catch_label.config(text=f"Total Catch (current trip): {trip.total_catch:.2f}kg")
                else:
                    # Dim unselected trips
                    for line in getattr(trip, "route_lines", []):
                        line.set_alpha(0.3)
                                  
    else:  # selected is a boat
        boat_num = int(selected[0])
        for boat in boat_objs:
            if boat.id == boat_num:
                for trip in boat.route:
                    for station in trip.nice_trip:
                        stations_in_trips.append(station)
                    for line in getattr(trip, "route_lines", []):
                        line.set_alpha(1.0)
            else:
                for trip in boat.route:
                    for line in getattr(trip, "route_lines", []):
                        line.set_alpha(0.3)
                    for sid in trip.nice_trip:
                        if sid in dict_station_num_annotations:
                            dict_station_num_annotations[sid].set_alpha(0.3)

    for key, annotation in dict_station_num_annotations.items():
        if key not in stations_in_trips:
            annotation.set_alpha(0.3)
        else: 
            annotation.set_alpha(1.0)

    canvas.draw_idle()


def add_trip_small(root, ax, canvas, boat_trip_tree, remaining_visited_tree):
    '''AI but works i think'''
    popup = tk.Toplevel(root)

    nice_label = tk.Label(popup, text="Enter Nice Trip:")
    nice_label.pack(pady=5)

    nice_entry = tk.Entry(popup)
    nice_entry.pack(pady=5)

    actual_label = tk.Label(popup, text="Enter Actual Trip:")
    actual_label.pack(pady=5)

    actual_entry = tk.Entry(popup)
    actual_entry.pack(pady=5)


    def submit():
        nice_input = nice_entry.get().strip()
        actual_input = actual_entry.get().strip()

        # Use whichever one is filled
        if nice_input:
            nice_input = eval(nice_input)
            if not nice_input[0] in ports_idx or not nice_input[-1] in ports_idx:
                return messagebox.showerror("Error", "First and Last nodes must be ports")
            
            for i in range(1, len(actual_input) - 1):
                if actual_input[i] not in station_ids:
                    return messagebox.showerror("Error", f'Node {actual_input[i]} is not a station')

            new_trip = Trip()
            new_trip.nice_trip = nice_input
            trip_objs.append(new_trip)

        # doesnt work atm?
        if actual_input:
            actual_input = eval(actual_input)
            if not actual_input[0] in ports_idx or not actual_input[-1] in ports_idx:
                return messagebox.showerror("Error", "First/Last node must be ports")
            
            if len(actual_input) <= 3:
                return

            nice_trip_from_actual = [actual_input[0]]
            for i in range(1, len(actual_input) - 2, 2):
                if actual_input[i] not in stations_idx:
                    return messagebox.showerror("Error", f'Node {actual_input[i]} is not a station')
                if abs(actual_input[i] - actual_input[i + 1]) != 1:
                    return messagebox.showerror("Error", f'Nodes {actual_input[i]} and {actual_input[i + 1]} are not a station pair')
                if actual_input[i] % 2 == 0:
                    diff_even = actual_input[i] - 14
                    diff_odd = actual_input[i + 1] - 13
                    if diff_even != diff_odd:
                        return messagebox.showerror("Error", f'Nodes {actual_input[i]} and {actual_input[i + 1]} are not a station pair')  
                    nice_trip_from_actual.append(int((actual_input[i] - 14)/2 + 13)) 
                if actual_input[i] % 2 == 1:
                    diff_even = actual_input[i] - 13
                    diff_odd = actual_input[i + 1] - 14
                    if diff_even != diff_odd:
                        return messagebox.showerror("Error", f'Nodes {actual_input[i]} and {actual_input[i + 1]} are not a station pair') 
                    nice_trip_from_actual.append(int((actual_input[i] - 13)/2 + 13))

            nice_trip_from_actual.append(actual_input[-1])

            new_trip = Trip()
            new_trip.actual_trip = actual_input
            new_trip.nice_trip = nice_trip_from_actual
            trip_objs.append(new_trip)
        
        update_tree_big(ax, canvas, boat_trip_tree, remaining_visited_tree)

        popup.destroy()

    submit_button = tk.Button(popup, text="Submit", command=submit)
    submit_button.pack(pady=10)

def on_close(fig, root):
    '''AI but should work'''
    plt.close(fig)
    root.destroy()
    sys.exit()

def heuristic_me(ax, canvas, root, boat_trip_tree, remaining_visited_tree):

    result = {"choice": None}

    def select_option(value):
        result["choice"] = value
        popup.destroy()

    popup = tk.Toplevel()
    popup.title("Choose an Option")

    tk.Label(popup, text="Choose an option:").pack(padx=30, pady=10)

    tk.Button(popup, text="next_descent_move", width=20, command=lambda: select_option("next")).pack(pady=5)
    tk.Button(popup, text="steepest_descent_move", width=20, command=lambda: select_option("steepest")).pack(pady=5)
    tk.Button(popup, text="next_descent_combined", width=20, command=lambda: select_option("combined")).pack(pady=5)

    # Wait until popup is closed
    popup.grab_set()
    popup.wait_window()

    stations_remaining = remaining_visited_tree.get_children("remain")
    stations_remaining = np.array(stations_remaining, dtype=int)
    stations_remaining -= 13
    stations_remaining_idx = np.stack([stations_remaining * 2, stations_remaining * 2 + 1]).T.reshape(-1) + n_ports

    Prob = Problem(list(stations_remaining_idx), ports_idx, fish_time_limit, n_boats)
    for boat in boat_objs:
        Prob.boats.append(boat)

    print(Prob.boats[0].route)
    if result["choice"] == "next":
        next_descent_move(Prob, k_catch, k_fish, upper_bound, work_limit, tsp=0.01, update_tree_big= lambda: update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree, True))
        print(Prob.boats[0].route)
    elif result["choice"] == "steepest":
        steepest_descent_move(Prob, k_catch, k_fish, upper_bound, work_limit, tsp=0.01, update_tree_big= lambda: update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree, True))
        print(Prob.boats[0].route)
    elif result["choice"] == "combined":
        next_descent_combined(Prob, k_catch, k_fish, upper_bound, work_limit, tsp=0.01, update_tree_big= lambda: update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree, True))
        print(Prob.boats[0].route)

    update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree)


def initial_solution_me(root, ax, canvas, boat_trip_tree, remaining_visited_tree):
    popup = tk.Toplevel(root)

    prob_num_label = tk.Label(popup, text="Test Problem Number (1-11):")
    prob_num_label.pack(pady=5)

    prob_num_entry = tk.Entry(popup)
    prob_num_entry.pack(pady=5)

    def init_sol_submit():
        prob_num_input = prob_num_entry.get().strip()
        
        try:
            prob_num = int(prob_num_input)
        except ValueError:
            return 

        test_ports_idx, test_stations_ids, _ = load_test_problem(prob_num)
        test_stations_idx = np.stack([test_stations_ids * 2, test_stations_ids * 2 + 1]).T.reshape(-1) + n_ports

        Prob = Problem(list(test_stations_idx), test_ports_idx, fish_time_limit, n_boats, boat_capacities, home_ports)
        Prob.generate_initial_solution()
        for boat in Prob.boats:
            for trip in boat.route:
                trip.actual_trip = trip.get_full_trip().copy()
                trip.actual_trip = [int(i) for i in trip.actual_trip]
                trip.get_nice_trip_from_actual_trip()
                trip_objs.append(trip)
        update_tree_big(ax, canvas, boat_trip_tree)
        popup.destroy()

    def whole_prob():

        Prob.generate_initial_solution()
        for boat in Prob.boats:
            for idx, trip in enumerate(boat.route):
                trip.id = f'trip:{boat.id}:{idx+1}'
                trip.actual_trip = trip.get_full_trip().copy()
                trip.actual_trip = [int(i) for i in trip.actual_trip]
                trip.get_nice_trip_from_actual_trip()
        update_tree_big(ax, canvas, root, boat_trip_tree, remaining_visited_tree)
        popup.destroy()

    submit_button = tk.Button(popup, text="Submit", command=init_sol_submit)
    submit_button.pack(pady=10)

    whole_prob_button = tk.Button(popup, text="Generate for Whole Problem",command=whole_prob)
    whole_prob_button.pack(pady=10)

def sort_children(boat_trip_tree, parent):
    children = boat_trip_tree.get_children(parent)
    sorted_children = sorted(
        children,
        key=lambda iid: int(boat_trip_tree.item(iid, "text"))
    )
    for idx, iid in enumerate(sorted_children):
        boat_trip_tree.move(iid, parent, idx)


'''ignore for now'''
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Auto-scroll

    def flush(self):
        pass  # Needed for compatibility


