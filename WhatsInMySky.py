import tkinter as tk
from tkinter import ttk, simpledialog
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import pandas as pd
import numpy as np
import os
import json
import threading
import webbrowser
import sys
import pytz

class WhatsInMySky:
    def __init__(self, root):
        self.root = root
        self.root.title("What's In My Sky v1.0 - Seti Astro")

        # Load previous settings
        self.settings_file = os.path.join(os.path.expanduser("~"), "sky_settings.json")
        self.settings = self.load_settings()

        # Input Fields
        self.latitude_label = tk.Label(root, text="Latitude:")
        self.latitude_label.grid(row=0, column=0, padx=5, pady=5)
        self.latitude_entry = tk.Entry(root)
        self.latitude_entry.grid(row=0, column=1, padx=5, pady=5)
        self.latitude_entry.insert(0, self.settings.get("latitude", ""))

        self.longitude_label = tk.Label(root, text="Longitude:")
        self.longitude_label.grid(row=1, column=0, padx=5, pady=5)
        self.longitude_entry = tk.Entry(root)
        self.longitude_entry.grid(row=1, column=1, padx=5, pady=5)
        self.longitude_entry.insert(0, self.settings.get("longitude", ""))

        self.date_label = tk.Label(root, text="Date (YYYY-MM-DD):")
        self.date_label.grid(row=2, column=0, padx=5, pady=5)
        self.date_entry = tk.Entry(root)
        self.date_entry.grid(row=2, column=1, padx=5, pady=5)
        self.date_entry.insert(0, self.settings.get("date", ""))

        self.time_label = tk.Label(root, text="Time (HH:MM):")
        self.time_label.grid(row=3, column=0, padx=5, pady=5)
        self.time_entry = tk.Entry(root)
        self.time_entry.grid(row=3, column=1, padx=5, pady=5)
        self.time_entry.insert(0, self.settings.get("time", ""))

        # Time Zone Dropdown
        self.timezone_label = tk.Label(root, text="Time Zone:")
        self.timezone_label.grid(row=4, column=0, padx=5, pady=5)
        self.timezone_combo = ttk.Combobox(root, values=pytz.all_timezones)
        self.timezone_combo.grid(row=4, column=1, padx=5, pady=5)
        self.timezone_combo.set(self.settings.get("timezone", "UTC"))

        # Catalog Filters
        self.catalog_filters_label = tk.Label(root, text="Catalog Filters:")
        self.catalog_filters_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.catalog_vars = {}
        catalogs = ["Messier", "NGC", "IC", "Caldwell", "Abell", "Sharpless"]
        for i, catalog in enumerate(catalogs):
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(root, text=catalog, variable=var)
            chk.grid(row=6 + (i // 2), column=i % 2, padx=5, pady=5, sticky='w')
            self.catalog_vars[catalog] = var

        # Button to calculate
        self.calculate_button = tk.Button(root, text="Calculate", command=self.start_calculation)
        self.calculate_button.grid(row=9, column=0, columnspan=2, pady=10)

        # Settings Icon for Object Limit
        self.settings_button = tk.Button(root, text=u"⚙", command=self.open_settings)  # Unicode for gear icon ⚙
        self.settings_button.grid(row=9, column=2, padx=5, pady=10)

        # Status Label
        self.status_label = tk.Label(root, text="Status: Idle")
        self.status_label.grid(row=10, column=0, columnspan=3, pady=5)

        # Local Sidereal Time Label
        self.lst_label = tk.Label(root, text="Local Sidereal Time: N/A")
        self.lst_label.grid(row=11, column=0, columnspan=3, pady=5)

        # Treeview to display results
        self.tree = ttk.Treeview(root, columns=("Name", "RA", "Dec", "Altitude", "Azimuth", "Minutes to Transit", "Before/After Transit", "Alt Name", "Type", "Magnitude", "Info"), show="headings")
        for col in self.tree['columns']:
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.tree, _col, False))
        self.tree.grid(row=12, column=0, columnspan=3, pady=10, sticky='nsew')

        # Add scrollbar to Treeview
        self.tree_scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=self.tree_scrollbar.set)
        self.tree_scrollbar.grid(row=12, column=3, sticky='ns', pady=10)

        # Configure column width
        for col in self.tree['columns']:
            self.tree.column(col, anchor='center', width=100)

        # Configure resizing
        root.grid_rowconfigure(12, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Object limit setting
        self.object_limit = self.settings.get("object_limit", 100)

        # Bind double-click event to Treeview
        self.tree.bind("<Double-1>", self.on_row_double_click)

        # Add custom object button
        self.add_object_button = tk.Button(root, text="Add Custom Object", command=self.add_custom_object)
        self.add_object_button.grid(row=9, column=3, padx=5, pady=10)

    def start_calculation(self):
        # Run the calculation in a separate thread to keep the GUI responsive
        calculation_thread = threading.Thread(target=self.calculate_objects)
        calculation_thread.start()

    def calculate_objects(self):
        try:
            # Update status label
            self.update_status("Calculating...")

            # Get user input
            latitude = float(self.latitude_entry.get())
            longitude = float(self.longitude_entry.get())
            date_str = self.date_entry.get()
            time_str = self.time_entry.get()
            timezone_str = self.timezone_combo.get()

            # Save settings
            self.save_settings(latitude, longitude, date_str, time_str, timezone_str)

            # Combine date and time
            datetime_str = f"{date_str} {time_str}"
            local = pytz.timezone(timezone_str)
            naive_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            local_datetime = local.localize(naive_datetime)
            astropy_time = Time(local_datetime)

            # Define observer's location
            location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)

            # Calculate Local Sidereal Time
            lst = astropy_time.sidereal_time('apparent', longitude * u.deg)
            print(f"Local Sidereal Time: {lst}")

            # Update LST label
            self.update_lst(f"Local Sidereal Time: {lst}")

            # Update status label
            self.update_status("Loading celestial catalog...")

            # Load celestial catalog from CSV
            catalog_file = os.path.join(os.path.dirname(sys.executable), "celestial_catalog.csv") if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "celestial_catalog.csv")

            if not os.path.exists(catalog_file):
                self.update_status("Catalog file not found.")
                return

            df = pd.read_csv(catalog_file, encoding='ISO-8859-1')

            if df.empty:
                self.update_status("Catalog file is empty.")
                return

            # Drop rows with invalid RA/Dec
            df.dropna(subset=['RA', 'Dec'], inplace=True)

            if df.empty:
                self.update_status("No valid celestial objects found in the catalog.")
                return

            # Apply catalog filters
            selected_catalogs = [catalog for catalog, var in self.catalog_vars.items() if var.get()]
            df = df[df['Catalog'].isin(selected_catalogs)]

            # Calculate Altitude and Azimuth for each object
            altaz_frame = AltAz(obstime=astropy_time, location=location)
            altitudes, azimuths, minutes_to_transit, before_after_transit = [], [], [], []

            for _, row in df.iterrows():
                sky_coord = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg, frame='icrs')
                altaz = sky_coord.transform_to(altaz_frame)
                altitudes.append(altaz.alt.deg)
                azimuths.append(altaz.az.deg)

                # Calculate time difference to transit
                ra = row['RA'] * u.deg.to(u.hourangle)  # Convert RA from degrees to hour angle
                time_diff = ((ra - lst.hour) * u.hour) % (24 * u.hour)
                minutes = time_diff.value * 60
                if minutes > 720:
                    minutes = 1440 - minutes
                    before_after_transit.append("After")
                else:
                    before_after_transit.append("Before")
                minutes_to_transit.append(minutes)

            df['Altitude'] = altitudes
            df['Azimuth'] = azimuths
            df['Minutes to Transit'] = minutes_to_transit
            df['Before/After Transit'] = before_after_transit

            # Filter out objects below the horizon
            df = df[df['Altitude'] > 0]

            if df.empty:
                self.update_status("No celestial objects are above the horizon at this time.")
                return

            # Sort by absolute minutes to transit
            df['Abs Minutes to Transit'] = df['Minutes to Transit'].abs()
            df = df.sort_values(by='Abs Minutes to Transit').drop(columns=['Abs Minutes to Transit'])

            # Limit results to top N (based on user setting)
            df = df.head(self.object_limit)

            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Populate the treeview with calculated data
            for _, row in df.iterrows():
                values = [
                    row['Name'],
                    row['RA'],
                    row['Dec'],
                    row['Altitude'],
                    row['Azimuth'],
                    row['Minutes to Transit'],
                    row['Before/After Transit'],
                    row.get('Alt Name', '') if pd.notna(row.get('Alt Name', '')) else '',
                    row.get('Type', '') if pd.notna(row.get('Type', '')) else '',
                    row.get('Magnitude', '') if pd.notna(row.get('Magnitude', '')) else '',
                    row.get('Info', '') if pd.notna(row.get('Info', '')) else ''
                ]
                self.tree.insert('', tk.END, values=values)

            # Update status label
            self.update_status("Calculation complete.")
        except ValueError as e:
            print(f"ValueError: {e}")
            self.update_status("Invalid input. Please check your latitude, longitude, date, and time.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.update_status("An unexpected error occurred. Please check the console for details.")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def update_lst(self, message):
        self.lst_label.config(text=message)

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        return {}

    def save_settings(self, latitude, longitude, date, time, timezone):
        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "date": date,
            "time": time,
            "timezone": timezone,
            "object_limit": self.object_limit
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)

    def open_settings(self):
        object_limit = simpledialog.askinteger("Settings", "Enter number of objects to display:", initialvalue=self.object_limit)
        if object_limit:
            self.object_limit = object_limit

    def treeview_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0]) if t[0] else float('inf'), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)

        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))

    def on_row_double_click(self, event):
        item = self.tree.selection()
        if item:
            item_values = self.tree.item(item, 'values')
            object_name = item_values[0].replace(" ", "")  # Remove spaces in the object name
            search_url = f"https://www.astrobin.com/search/?q={object_name}"
            webbrowser.open(search_url)

    def add_custom_object(self):
        name = simpledialog.askstring("Add Custom Object", "Enter object name:")
        ra = simpledialog.askfloat("Add Custom Object", "Enter RA (in degrees):")
        dec = simpledialog.askfloat("Add Custom Object", "Enter Dec (in degrees):")

        if name and ra is not None and dec is not None:
            new_object = {
                "Name": name,
                "RA": ra,
                "Dec": dec,
                "Catalog": "User Defined",
                "Alt Name": "User Defined",
                "Type": "Custom",
                "Magnitude": "",
                "Info": ""
            }
            catalog_file = os.path.join(os.path.dirname(sys.executable), "celestial_catalog.csv") if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "celestial_catalog.csv")
            df = pd.read_csv(catalog_file, encoding='ISO-8859-1')
            df = pd.concat([df, pd.DataFrame([new_object])], ignore_index=True)
            df.to_csv(catalog_file, index=False, encoding='ISO-8859-1')
            self.update_status(f"Added custom object: {name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhatsInMySky(root)
    root.mainloop()

# Franklin Marek
# www.setiastro.com
# Copyright 2024
