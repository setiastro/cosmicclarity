import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
import pandas as pd
import numpy as np
import os
import json
import threading
import webbrowser
import sys
import pytz
from PIL import Image, ImageTk
from astropy.coordinates import get_sun
import warnings
from decimal import Decimal, getcontext

# Set precision for Decimal operations
getcontext().prec = 24

# Suppress warnings
warnings.filterwarnings("ignore")

class WhatsInMySky:
    def __init__(self, root):
        self.root = root
        self.root.title("What's In My Sky v1.2 - Seti Astro")

        # Load previous settings
        self.settings_file = os.path.join(os.path.expanduser("~"), "sky_settings.json")
        self.settings = self.load_settings()

        # Input Fields
        self.latitude_label = tk.Label(root, text="Latitude:")
        self.latitude_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.latitude_entry = tk.Entry(root)
        self.latitude_entry.grid(row=0, column=1, padx=5, pady=5)
        self.latitude_entry.insert(0, self.settings.get("latitude", ""))

        self.longitude_label = tk.Label(root, text="Longitude:")
        self.longitude_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.longitude_entry = tk.Entry(root)
        self.longitude_entry.grid(row=1, column=1, padx=5, pady=5)
        self.longitude_entry.insert(0, self.settings.get("longitude", ""))

        self.date_label = tk.Label(root, text="Date (YYYY-MM-DD):")
        self.date_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.date_entry = tk.Entry(root)
        self.date_entry.grid(row=2, column=1, padx=5, pady=5)
        self.date_entry.insert(0, self.settings.get("date", ""))

        self.time_label = tk.Label(root, text="Time (HH:MM):")
        self.time_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.time_entry = tk.Entry(root)
        self.time_entry.grid(row=3, column=1, padx=5, pady=5)
        self.time_entry.insert(0, self.settings.get("time", ""))

        # Time Zone Dropdown
        self.timezone_label = tk.Label(root, text="Time Zone:")
        self.timezone_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')
        self.timezone_combo = ttk.Combobox(root, values=pytz.all_timezones)
        self.timezone_combo.grid(row=4, column=1, padx=5, pady=5)
        self.timezone_combo.set(self.settings.get("timezone", "UTC"))

        # Catalog Filters Frame
        self.catalog_filters_label = tk.Label(root, text="Catalog Filters:")
        self.catalog_filters_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        
        self.catalog_frame = tk.Frame(root)
        self.catalog_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.catalog_vars = {}
        catalogs = ["Messier", "NGC", "IC", "Caldwell", "Abell", "Sharpless", "LBN", "LDN", "PNG", "User"]
        for i, catalog in enumerate(catalogs):
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(self.catalog_frame, text=catalog, variable=var)
            chk.grid(row=i // 5, column=i % 5, padx=5, pady=5, sticky='w')
            self.catalog_vars[catalog] = var

        # RA/Dec Format Frame
        self.ra_dec_frame = tk.Frame(root)
        self.ra_dec_frame.grid(row=11, column=0, columnspan=3, pady=10)
        
        self.ra_dec_format = tk.StringVar(value="Degrees")
        self.ra_dec_label = tk.Label(self.ra_dec_frame, text="RA/Dec Format:")
        self.ra_dec_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ra_dec_degrees = tk.Radiobutton(self.ra_dec_frame, text="Degrees", variable=self.ra_dec_format, value="Degrees")
        self.ra_dec_degrees.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.ra_dec_hms = tk.Radiobutton(self.ra_dec_frame, text="H:M:S / D:M:S", variable=self.ra_dec_format, value="HMS")
        self.ra_dec_hms.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Add trace to RA/Dec format to trigger updates
        self.ra_dec_format.trace_add("write", self.update_ra_dec_format)

        # Button to calculate
        self.calculate_button = tk.Button(root, text="Calculate", command=self.start_calculation)
        self.calculate_button.grid(row=10, column=0, columnspan=2, pady=10)

        # Settings Icon for Object Limit
        self.settings_button = tk.Button(root, text=u"⚙", command=self.open_settings)  # Unicode for gear icon ⚙
        self.settings_button.grid(row=10, column=2, padx=5, pady=10)

        # Status Label
        self.status_label = tk.Label(root, text="Status: Idle")
        self.status_label.grid(row=12, column=0, columnspan=3, pady=5)

        # Local Sidereal Time Label
        self.lst_label = tk.Label(root, text="Local Sidereal Time: {:.3f}".format(0.0))
        self.lst_label.grid(row=13, column=0, columnspan=3, pady=5)

        # Lunar Phase Image and Label
        self.lunar_phase_image_label = tk.Label(root)
        self.lunar_phase_image_label.grid(row=0, column=3, rowspan=4, padx=5, pady=5, sticky='ne')
        self.lunar_phase_label = tk.Label(root, text="Lunar Phase: N/A")
        self.lunar_phase_label.grid(row=4, column=3, padx=5, pady=5)

        # Treeview to display results
        self.tree = ttk.Treeview(root, columns=("Name", "RA", "Dec", "Altitude", "Azimuth", "Minutes to Transit", "Before/After Transit", "Degrees from Moon", "Alt Name", "Type", "Magnitude", "Size (arcmin)"), show="headings")
        for col in self.tree['columns']:
            self.tree.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(self.tree, _col, False))
        self.tree.grid(row=14, column=0, columnspan=3, pady=10, sticky='nsew')

        # Add scrollbar to Treeview
        self.tree_scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=self.tree_scrollbar.set)
        self.tree_scrollbar.grid(row=14, column=3, sticky='ns', pady=10)

        # Configure column width
        for col in self.tree['columns']:
            self.tree.column(col, anchor='center', width=100)

        # Configure resizing
        root.grid_rowconfigure(14, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Object limit setting
        self.object_limit = self.settings.get("object_limit", 100)

        # Bind double-click event to Treeview
        self.tree.bind("<Double-1>", self.on_row_double_click)

        # Add custom object button
        self.add_object_button = tk.Button(root, text="Add Custom Object", command=self.add_custom_object)
        self.add_object_button.grid(row=10, column=3, padx=5, pady=10)

        # Add save button to export Treeview data to CSV
        self.save_button = tk.Button(root, text="Save to CSV", command=self.save_to_csv)
        self.save_button.grid(row=11, column=3, padx=5, pady=10)

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

            # Calculate lunar phase
            self.calculate_lunar_phase(astropy_time, location)

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

            # Calculate Altitude, Azimuth, and Degrees from Moon for each object
            altaz_frame = AltAz(obstime=astropy_time, location=location)
            altitudes, azimuths, minutes_to_transit, before_after_transit, degrees_from_moon = [], [], [], [], []

            moon = get_body("moon", astropy_time, location).transform_to(altaz_frame)

            for _, row in df.iterrows():
                sky_coord = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg, frame='icrs')
                altaz = sky_coord.transform_to(altaz_frame)
                altitudes.append(round(altaz.alt.deg, 1))
                azimuths.append(round(altaz.az.deg, 1))

                # Calculate time difference to transit
                ra = Decimal(row['RA']) * Decimal(u.deg.to(u.hourangle))  # Convert RA from degrees to hour angle
                time_diff = (ra - Decimal(lst.hour)) % Decimal(24)
                if time_diff < 0:
                    before_after_transit.append("After")
                else:
                    before_after_transit.append("Before")
                minutes = round(abs(time_diff) * Decimal(60))
                minutes_to_transit.append(minutes)

                # Calculate angular distance from the moon
                moon_sep = sky_coord.separation(moon).deg
                degrees_from_moon.append(round(moon_sep, 2))

            df['Altitude'] = altitudes
            df['Azimuth'] = azimuths
            df['Minutes to Transit'] = minutes_to_transit
            df['Before/After Transit'] = before_after_transit
            df['Degrees from Moon'] = degrees_from_moon

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
                ra_display = row['RA']
                dec_display = row['Dec']

                if self.ra_dec_format.get() == "HMS":
                    ra_display = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg).ra.to_string(unit=u.hour, sep=':')
                    dec_display = SkyCoord(ra=row['RA'] * u.deg, dec=row['Dec'] * u.deg).dec.to_string(unit=u.deg, sep=':')

                values = [
                    row['Name'],
                    ra_display,
                    dec_display,
                    row['Altitude'],
                    row['Azimuth'],
                    row['Minutes to Transit'],
                    row['Before/After Transit'],
                    row['Degrees from Moon'],
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

    def calculate_lunar_phase(self, astropy_time, location):
        moon = get_body("moon", astropy_time, location)
        sun = get_sun(astropy_time)
        elongation = moon.separation(sun).deg

        # Determine lunar phase percentage
        phase_percentage = (1 - np.cos(np.radians(elongation))) / 2 * 100
        phase_percentage = round(phase_percentage)

        # Select appropriate lunar phase image based on phase angle
        phase_folder = os.path.join(sys._MEIPASS, "imgs") if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "imgs")
        phase_image_name = "new_moon.png"  # Default

        if 0 <= elongation < 9:
            phase_image_name = "new_moon.png"
        elif 9 <= elongation < 18:
            phase_image_name = "waxing_crescent_1.png"
        elif 18 <= elongation < 27:
            phase_image_name = "waxing_crescent_2.png"
        elif 27 <= elongation < 36:
            phase_image_name = "waxing_crescent_3.png"
        elif 36 <= elongation < 45:
            phase_image_name = "waxing_crescent_4.png"
        elif 45 <= elongation < 54:
            phase_image_name = "waxing_crescent_5.png"
        elif 54 <= elongation < 90:
            phase_image_name = "first_quarter.png"
        elif 90 <= elongation < 108:
            phase_image_name = "waxing_gibbous_1.png"
        elif 108 <= elongation < 126:
            phase_image_name = "waxing_gibbous_2.png"
        elif 126 <= elongation < 144:
            phase_image_name = "waxing_gibbous_3.png"
        elif 144 <= elongation < 162:
            phase_image_name = "waxing_gibbous_4.png"
        elif 162 <= elongation < 180:
            phase_image_name = "full_moon.png"
        elif 180 <= elongation < 198:
            phase_image_name = "waning_gibbous_1.png"
        elif 198 <= elongation < 216:
            phase_image_name = "waning_gibbous_2.png"
        elif 216 <= elongation < 234:
            phase_image_name = "waning_gibbous_3.png"
        elif 234 <= elongation < 252:
            phase_image_name = "waning_gibbous_4.png"
        elif 252 <= elongation < 270:
            phase_image_name = "last_quarter.png"
        elif 270 <= elongation < 279:
            phase_image_name = "waning_crescent_1.png"
        elif 279 <= elongation < 288:
            phase_image_name = "waning_crescent_2.png"
        elif 288 <= elongation < 297:
            phase_image_name = "waning_crescent_3.png"
        elif 297 <= elongation < 306:
            phase_image_name = "waning_crescent_4.png"
        elif 306 <= elongation < 315:
            phase_image_name = "waning_crescent_5.png"

        # Load and display the lunar phase image
        phase_image_path = os.path.join(phase_folder, phase_image_name)
        if os.path.exists(phase_image_path):
            image = Image.open(phase_image_path)
            image = image.resize((100, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.lunar_phase_image_label.config(image=photo)
            self.lunar_phase_image_label.image = photo

        # Update lunar phase label
        self.lunar_phase_label.config(text=f"Lunar Phase: {phase_percentage}% illuminated")

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

    def update_ra_dec_format(self, *args):
        for item in self.tree.get_children():
            row = self.tree.item(item, 'values')
            ra_value = row[1]
            dec_value = row[2]

            try:
                if self.ra_dec_format.get() == "HMS":
                    ra_deg = float(ra_value)
                    dec_deg = float(dec_value)
                    sky_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
                    ra_display = sky_coord.ra.to_string(unit=u.hour, sep=':')
                    dec_display = sky_coord.dec.to_string(unit=u.deg, sep=':')
                else:
                    ra_hms = SkyCoord(ra=ra_value, dec=dec_value, unit=(u.hourangle, u.deg))
                    ra_display = round(ra_hms.ra.deg, 3)
                    dec_display = round(ra_hms.dec.deg, 3)
            except Exception as e:
                # Handle invalid conversion and print the error for debugging
                print(f"Conversion error: {e}")
                ra_display = ra_value
                dec_display = dec_value

            values = list(row)
            values[1] = ra_display
            values[2] = dec_display
            self.tree.item(item, values=values)


    def save_to_csv(self):
        # Ask user where to save the CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            # Extract data from Treeview
            columns = self.tree['columns']
            data = [columns]
            for item in self.tree.get_children():
                data.append(self.tree.item(item, 'values'))

            # Convert data to DataFrame and save as CSV
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(file_path, index=False)
            self.update_status(f"Data saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhatsInMySky(root)
    root.mainloop()

# Franklin Marek
# www.setiastro.com
# Copyright 2024
