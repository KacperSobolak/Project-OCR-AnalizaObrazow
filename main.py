import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import segmentation as s
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

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

        self.process_button = ttk.Button(self.buttons_frame, text="Przetwórz zdjęcie", command=self.handleProcessButton)
        self.process_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Ramka do wyświetlania wyników OCRa
        self.results_frame = ttk.LabelFrame(self, text="Wyniki", padding=(10, 10))
        self.results_frame.grid(
            row=1, column=0, padx=10, pady=(10, 10), sticky="nsew"
        )
        self.results_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(self.results_frame, text="Proces OCR:\n")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.results_label = ttk.Label(self.results_frame, text="Odczytany tekst rejestracji:\n")
        self.results_label.grid(row=1, column=0, sticky="w")

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

    def handleProcessButton(self):
        self.status_label.config(text="Proces OCR:\nROZPOCZĘTY")
        self.process_image_with_error_handling()

    def process_image_with_error_handling(self):
        try:
            self.process_image()
        except Exception as e:
            self.status_label.config(text="Proces OCR:\nBŁĄD")
            return

    @staticmethod
    def get_file_path():
        app = QApplication(sys.argv)
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Wybierz obraz",
            "",
            "Image Files (*.png *.jpg)",
            options=options
        )
        return file_path

    def load_image(self):
        file_path = self.get_file_path()

        if file_path:
            self.image = Image.open(file_path)

            self.instruction_label.grid_forget()
            if self.results_label:
                self.results_label.config(text="Odczytany tekst rejestracji:\n")
            if self.status_label:
                self.status_label.config(text="Proces OCR:\n")

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
        
        plt.figure()

        # Reformat załadowanego obrazu przez Pillow do cv2
        img = np.array(self.image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray=img.mean(axis=2)/255

        v_edges=s.vertical_sobel(gray)

        v_proj=s.vertical_projection(v_edges)

        v_proj_blured=s.blur_vector(v_proj,np.ones(9)/9)

        v_bands=s.band_detection(v_proj_blured.copy())

        candidates=[]
        for v_band in v_bands:
            blur_size = img.shape[1] // 5
            blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
            h_edges=s.horizontal_sobel(gray[v_band[0]:v_band[1],:])
            h_proj=s.horizontal_projection(h_edges)
            h_proj_blured=s.blur_vector(h_proj,np.ones(blur_size)/blur_size)
            h_bands=s.band_detection(h_proj_blured.copy())
            for h_band in h_bands:
                h_band_proj=h_proj_blured[h_band[0]:h_band[1]]
                deriv=s.derivative(h_band_proj)
                h_range=s.analyze_derivative(deriv)
                projection=v_proj[v_band[0]:v_band[1]]
                candidates.append([v_band,[h_band[0]+h_range[0],h_band[0]+h_range[1]],s.heuristic(img,gray,[h_band[0]+h_range[0],h_band[0]+h_range[1]],v_band)])

        sorted_candidates = sorted(candidates, key=lambda x: x[2])

        matches = []
        max = 0
        for c in sorted_candidates:
            pom_l=c[1][0]-15
            if(pom_l<0):
                pom_l=0
            pom_r=c[1][1]+15
            if(pom_r>gray.shape[1]):
                pom_r=gray.shape[1]
            image,thresh=s.preprocess_image(img[c[0][0]:c[0][1],pom_l:pom_r])
            contours=s.find_contours(thresh)
            sorted_characters=s.extract_characters(thresh,contours)
            if max < len(sorted_characters):
                max = len(sorted_characters)
                matches.clear()
                matches = sorted_characters
        
        text=s.get_text_from_character_images(matches)
        self.results_label.config(text="Odczytany tekst rejestracji:\n" + text)
        self.status_label.config(text="Proces OCR:\nZAKOŃCZONY")


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