import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class OCRApp(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        self.columnconfigure(0, minsize=300)
        self.columnconfigure(1, weight=1)

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        self.image = None
        self.setup_widgets()

    def setup_widgets(self):
        # Ramka do przycisków ładujących i przetwarzających obraz
        self.buttons_frame = ttk.LabelFrame(self, text="Opcje", padding=(10, 10))
        self.buttons_frame.grid(
            row=0, column=0, padx=10, pady=(10, 10), sticky="nsew"
        )
        self.buttons_frame.columnconfigure(0, weight=1)

        self.load_button = ttk.Button(self.buttons_frame, text="Wczytaj zdjęcie", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.process_button = ttk.Button(self.buttons_frame, text="Przetwórz zdjęcie", command=self.process_image)
        self.process_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Ramka do wyświetlania wyników OCRa
        self.results_frame = ttk.LabelFrame(self, text="Wyniki", padding=(10, 10))
        self.results_frame.grid(
            row=1, column=0, padx=10, pady=(10, 10), sticky="nsew"
        )
        self.results_frame.columnconfigure(0, weight=1)

        self.results_label = None

        # Ramka z przyciskami pomocy (jak użyć / informacja o autorach)
        self.help_frame = ttk.LabelFrame(self, text="Pomoc", padding=(10, 10))
        self.help_frame.grid(
            row=2, column=0, padx=10, pady=(10, 10), sticky="nsew"
        )
        self.help_frame.columnconfigure(0, weight=1)

        self.help_button = ttk.Button(self.help_frame, text="Jak używać", command=self.show_how_to_use)
        self.help_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.authors_button = ttk.Button(self.help_frame, text="O autorach", command=self.show_authors)
        self.authors_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Ramka do wyświetlania procesowanego obrazu
        self.image_frame = ttk.LabelFrame(self, text="Podgląd", padding=(10, 10))
        self.image_frame.grid(
            row=0, column=1, padx=10, pady=(10, 10), sticky="nsew", rowspan=3
        )

        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.instruction_label = tk.Label(self.image_frame, text="Wczytaj zdjęcie aby rozpocząć.", anchor="center")
        self.instruction_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.progress_bar = ttk.Progressbar(self.image_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.progress_bar['value'] = 0

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

        if file_path:
            self.progress_bar["value"] = 0
            self.image = Image.open(file_path)

            self.instruction_label.grid_forget()
            if self.results_label:
                self.results_label.grid_forget()

            self.display_image()

    def display_image(self):
        thumbnail_image = self.image
        thumbnail_image.thumbnail((self.image_frame.winfo_width(), self.image_frame.winfo_height()))

        self.tk_image = ImageTk.PhotoImage(thumbnail_image)

        self.image_label = tk.Label(self.image_frame, image=self.tk_image)
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.image_label.image = self.tk_image

    def process_image(self):
        if self.image == None:
            messagebox.showwarning("Brak zdjęcia do przetworzenia", "Wczytaj zdjęcie przed przetworzeniem!")
            return
        
        self.progress_bar["value"] = 0
        self.update_progress(0)

    def update_progress(self, value):
        if value <= 100:
            self.progress_bar["value"] = value
            self.after(50, self.update_progress, value + 5)
        else:
            self.results_label = tk.Label(self.results_frame, text="Odczytana rejestracja:\n\nWKZNY68", anchor="w", justify="left")
            self.results_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def show_how_to_use(self):
        messagebox.showinfo("Jak używać", "Jak używać aplikacji:\n\n"
        "1. Kliknij 'Wczytaj zdjęcie' aby wybrać obraz.\n"
        "2. Kliknij 'Przetwórz zdjęcie' aby rozpocząć proces OCR.\n"
        "3. Wyniki pojawią się w sekcji 'Wyniki'.\n")

    def show_authors(self):
        messagebox.showinfo("O autorach", "Aplikacja stworzona przez:\n\n"
        "- Joanna Hełdak\n"
        "- Kacper Sobolak\n"
        "- Szymon Konieczny\n"
        "- Tymoteusz Kałuzieński")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("OCR Rejestracji Samochodowych")

    root.geometry("800x600")
    root.resizable(False, False)

    root.tk.call("source", "azure.tcl")
    root.tk.call("set_theme", "dark")

    app = OCRApp(root)
    app.pack(fill="both", expand=True)

    root.mainloop()