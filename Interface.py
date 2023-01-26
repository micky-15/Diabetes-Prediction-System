from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QGridLayout
from PyQt5.QtCore import Qt
import sys
import pandas as pd

import model
def userInterface() :
    model.prediction_model()

    class UI(QWidget):
        def __init__(self):
            super().__init__()
            self.resize(400, 400)
            self.setWindowTitle("Diabetes Detection System")
            self.setContentsMargins(50, 50, 50, 50)

            layout = QGridLayout()
            self.setLayout(layout)

            self.label1 = QLabel("Pregnencies : ")
            layout.addWidget(self.label1, 0, 0)

            self.label2 = QLabel("Blood Pressure : ")
            layout.addWidget(self.label2, 1, 0)

            self.label3 = QLabel("Glucose : ")
            layout.addWidget(self.label3, 2, 0)

            self.label4 = QLabel("Skin Thickness : ")
            layout.addWidget(self.label4, 3, 0)

            self.label5 = QLabel("Insulin level : ")
            layout.addWidget(self.label5, 4, 0)

            self.label6 = QLabel("BMI : ")
            layout.addWidget(self.label6, 5, 0)

            self.label7 = QLabel("Diabetes Pedigree Function : ")
            layout.addWidget(self.label7, 6, 0)

            self.label8 = QLabel("Age : ")
            layout.addWidget(self.label8, 7, 0)

            self.label9 = QLabel("Prediction : ")
            layout.addWidget(self.label9, 9, 0)

            self.input1 = QLineEdit()  # pregnencies
            layout.addWidget(self.input1, 0, 1)

            self.input2 = QLineEdit()  # bp
            layout.addWidget(self.input2, 1, 1)

            self.input3 = QLineEdit()  # Glucose
            layout.addWidget(self.input3, 2, 1)

            self.input4 = QLineEdit()  # SkinThickness
            layout.addWidget(self.input4, 3, 1)

            self.input5 = QLineEdit()  # insulin level
            layout.addWidget(self.input5, 4, 1)

            self.input6 = QLineEdit()  # bmi
            layout.addWidget(self.input6, 5, 1)

            self.input7 = QLineEdit()  # Diabetes Pedigree function
            layout.addWidget(self.input7, 6, 1)

            self.input8 = QLineEdit()  # Age
            layout.addWidget(self.input8, 7, 1)

            # self.outputbox = QLabel("Prediction ")
            # layout.addWidget(self.outputbox, 8, 0)

            self.displaybox = QLabel()
            layout.addWidget(self.displaybox, 9, 1)

            button = QPushButton("Submit")
            button.setFixedWidth(50)
            button.clicked.connect(self.disp)
            layout.addWidget(button, 8, 1, Qt.AlignmentFlag.AlignRight)

        def disp(self):
            preg = self.input1.text()
            bp = self.input2.text()
            gluc = self.input3.text()
            skin_thick = self.input4.text()
            insu = self.input5.text()
            bmi = self.input6.text()
            DPF = self.input7.text()
            AGE = self.input8.text()
            d = {'Pregnancies': int(preg), 'Glucose': int(bp), 'BloodPressure': int(gluc), 'SkinThickness': int(skin_thick), 'Insulin': int(insu), 'BMI': int(bmi), 'DiabetesPedigreeFunction': int(DPF), 'Age': int(AGE)}
            
            e = pd.DataFrame(data=d, index=[1])
            z = model.predict(e)
            if z == 1:

                self.displaybox.setText("Diabetic")
                # Diabetic

            else:
                self.displaybox.setText("Not Diabetic")

            # print(f"pregnencies = {preg} ")
            # print(f"bp = {bp} ")
            # print(f"gluc = {gluc} ")
            # print(f"skin_thick = {skin_thick} ")
            # print(f"insu = {insu} ")
            # print(f"bmi = {bmi} ")
            # print(f"DPF = {DPF} ")
            # print(f"AGE = {AGE} ")
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec())            
